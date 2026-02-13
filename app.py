
import os
import json
import uuid
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, Date, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from sqlalchemy.exc import IntegrityError

from passlib.context import CryptContext
from jose import jwt, JWTError

# ==========================
# Config
# ==========================
APP_NAME = "GouravNxMx Insurance Intelligence Suite (GIIS)"
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # default 7 days

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./giis.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def now_utc() -> datetime:
    return datetime.utcnow()


def gen_uuid() -> str:
    return str(uuid.uuid4())


# ==========================
# DB Models
# ==========================
class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String, primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    country = Column(String, nullable=False, default="IN")  # IN / US
    timezone = Column(String, nullable=False, default="Asia/Kolkata")
    created_at = Column(DateTime, default=now_utc)

    users = relationship("User", back_populates="tenant", cascade="all, delete-orphan")
    clients = relationship("Client", back_populates="tenant", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=False)
    role = Column(String, nullable=False, default="agent")  # admin / agent
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=now_utc)

    tenant = relationship("Tenant", back_populates="users")


class FeatureFlag(Base):
    __tablename__ = "feature_flags"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=False)
    key = Column(String, index=True, nullable=False)
    enabled = Column(Integer, nullable=False, default=1)  # 0/1
    limits_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_utc)


class Client(Base):
    __tablename__ = "clients"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, ForeignKey("tenants.id"), index=True, nullable=False)

    name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)

    age = Column(Integer, nullable=True)
    income_band = Column(String, nullable=True)
    city = Column(String, nullable=True)
    language_pref = Column(String, nullable=False, default="en")  # en/hi/mr

    engagement_score = Column(Float, nullable=False, default=50.0)
    risk_score = Column(Float, nullable=False, default=50.0)

    created_at = Column(DateTime, default=now_utc)

    tenant = relationship("Tenant", back_populates="clients")
    policies = relationship("Policy", back_populates="client", cascade="all, delete-orphan")
    interactions = relationship("Interaction", back_populates="client", cascade="all, delete-orphan")


class Policy(Base):
    __tablename__ = "policies"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, index=True, nullable=False)
    client_id = Column(String, ForeignKey("clients.id"), index=True, nullable=False)

    carrier = Column(String, nullable=True)
    policy_type = Column(String, nullable=False, default="term")
    premium_amount = Column(Float, nullable=False, default=0.0)
    premium_frequency = Column(String, nullable=False, default="yearly")  # monthly/quarterly/yearly
    start_date = Column(Date, nullable=True)
    maturity_date = Column(Date, nullable=True)
    sum_assured = Column(Float, nullable=True)
    commission_rate = Column(Float, nullable=False, default=0.05)
    status = Column(String, nullable=False, default="active")  # active/lapsed/matured
    created_at = Column(DateTime, default=now_utc)

    client = relationship("Client", back_populates="policies")


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, index=True, nullable=False)
    client_id = Column(String, ForeignKey("clients.id"), index=True, nullable=False)
    user_id = Column(String, index=True, nullable=True)

    channel = Column(String, nullable=False, default="meeting")  # call/meeting/whatsapp/email
    occurred_at = Column(DateTime, nullable=False, default=now_utc)
    notes = Column(Text, nullable=True)
    sentiment_score = Column(Float, nullable=True)  # -1..1
    objection_tags_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=now_utc)

    client = relationship("Client", back_populates="interactions")


class AIInsight(Base):
    __tablename__ = "ai_insights"
    id = Column(String, primary_key=True, default=gen_uuid)
    tenant_id = Column(String, index=True, nullable=False)
    client_id = Column(String, index=True, nullable=True)
    insight_type = Column(String, index=True, nullable=False)
    score = Column(Float, nullable=False, default=0.0)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=now_utc)


Base.metadata.create_all(bind=engine)

# ==========================
# Pydantic Schemas
# ==========================
class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TenantOut(BaseModel):
    id: str
    name: str
    country: str
    timezone: str


class MeOut(BaseModel):
    id: str
    tenant: TenantOut
    role: str
    name: str
    email: EmailStr


class ClientIn(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[EmailStr] = None
    age: Optional[int] = None
    income_band: Optional[str] = None
    city: Optional[str] = None
    language_pref: str = "en"


class ClientOut(BaseModel):
    id: str
    name: str
    phone: Optional[str]
    email: Optional[str]
    age: Optional[int]
    income_band: Optional[str]
    city: Optional[str]
    language_pref: str
    engagement_score: float
    risk_score: float
    created_at: datetime


class PolicyIn(BaseModel):
    client_id: str
    carrier: Optional[str] = None
    policy_type: str = "term"
    premium_amount: float = 0.0
    premium_frequency: str = "yearly"
    start_date: Optional[date] = None
    maturity_date: Optional[date] = None
    sum_assured: Optional[float] = None
    commission_rate: float = 0.05
    status: str = "active"


class PolicyOut(BaseModel):
    id: str
    client_id: str
    carrier: Optional[str]
    policy_type: str
    premium_amount: float
    premium_frequency: str
    start_date: Optional[date]
    maturity_date: Optional[date]
    sum_assured: Optional[float]
    commission_rate: float
    status: str
    created_at: datetime


class InteractionIn(BaseModel):
    client_id: str
    channel: str = "meeting"
    occurred_at: Optional[datetime] = None
    notes: Optional[str] = None
    sentiment_score: Optional[float] = None
    objection_tags: Optional[List[str]] = None


class ForecastOut(BaseModel):
    months: List[str]
    expected_commission: List[float]
    expected_premium: List[float]


class HeatmapRow(BaseModel):
    client_id: str
    client_name: str
    risk_score: float
    bucket: str
    reasons: List[str]


class FollowupRow(BaseModel):
    client_id: str
    client_name: str
    followup_score: float
    bucket: str
    reasons: List[str]


# ==========================
# Auth helpers
# ==========================
import bcrypt

def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw, salt).decode("utf-8")

def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )
    except Exception:
        return False

def create_access_token(data: dict, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    to_encode = dict(data)
    exp = datetime.utcnow() + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": exp})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ==========================
# Scoring engines (v1 rules)
# ==========================
def clamp(a: float, b: float, x: float) -> float:
    return max(a, min(b, x))


def days_between(d1: datetime, d2: datetime) -> int:
    return abs((d2 - d1).days)


def followup_score_for_client(db: Session, tenant_id: str, client: Client) -> Dict[str, Any]:
    reasons = []
    score = 0.0

    last = db.query(Interaction).filter(
        Interaction.tenant_id == tenant_id,
        Interaction.client_id == client.id
    ).order_by(Interaction.occurred_at.desc()).first()

    if last:
        d = days_between(last.occurred_at, now_utc())
        s = clamp(0, 20, (d / 30.0) * 20.0)
        score += s
        reasons.append(f"Last contact {d} day(s) ago (+{s:.0f})")
        if last.sentiment_score is not None:
            sentiment = clamp(-1, 1, last.sentiment_score)
            ss = sentiment * 10.0
            score += ss
            reasons.append(f"Last sentiment {sentiment:+.2f} ({ss:+.0f})")
    else:
        score += 12
        reasons.append("No interactions yet (+12)")

    today = date.today()
    policies = db.query(Policy).filter(
        Policy.tenant_id == tenant_id,
        Policy.client_id == client.id,
        Policy.status == "active"
    ).all()

    maturity_soon = 0
    premium_value = 0.0
    for p in policies:
        premium_value += float(p.premium_amount or 0.0)
        if p.maturity_date:
            days_to_maturity = (p.maturity_date - today).days
            if 0 <= days_to_maturity <= 90:
                maturity_soon += 1

    if maturity_soon > 0:
        score += 20
        reasons.append(f"{maturity_soon} policy(ies) mature within 90 days (+20)")

    due_soon = any((p.premium_frequency or "").lower() in ["monthly", "quarterly"] for p in policies)
    if due_soon:
        score += 25
        reasons.append("Premium cycles are frequent (monthly/quarterly) (+25)")

    score += clamp(0, 15, (client.engagement_score / 100.0) * 15.0)
    reasons.append(f"Engagement score {client.engagement_score:.0f} (+{(client.engagement_score/100)*15:.0f})")

    score += clamp(0, 10, (premium_value / 50000.0) * 10.0)
    reasons.append(f"Premium value impact (+{clamp(0,10,(premium_value/50000.0)*10.0):.0f})")

    score = clamp(0, 100, score)
    bucket = "high" if score >= 70 else ("medium" if score >= 40 else "low")
    return {"score": score, "bucket": bucket, "reasons": reasons}


def risk_score_for_client(db: Session, tenant_id: str, client: Client) -> Dict[str, Any]:
    reasons = []
    score = 0.0

    policies = db.query(Policy).filter(
        Policy.tenant_id == tenant_id,
        Policy.client_id == client.id,
        Policy.status == "active"
    ).all()

    if not policies:
        score += 35
        reasons.append("No active policies (portfolio unknown) (+35)")
    else:
        types = {}
        for p in policies:
            t = (p.policy_type or "other").lower()
            types[t] = types.get(t, 0) + 1
        top_share = max(types.values()) / max(1, len(policies))
        conc = clamp(0, 25, (top_share - 0.5) * 50)
        if conc > 0:
            score += conc
            reasons.append(f"Policy type concentration ({top_share:.0%}) (+{conc:.0f})")

        if "health" not in types:
            score += 15
            reasons.append("No health cover detected (+15)")

        today = date.today()
        maturities_90 = 0
        for p in policies:
            if p.maturity_date:
                d = (p.maturity_date - today).days
                if 0 <= d <= 90:
                    maturities_90 += 1
        if maturities_90 >= 2:
            score += 15
            reasons.append(f"{maturities_90} maturities within 90 days (+15)")

        income = (client.income_band or "").strip()
        sa_total = sum(float(p.sum_assured or 0.0) for p in policies if p.sum_assured is not None)
        income_target = {
            "<5L": 2000000,
            "5-10L": 5000000,
            "10-25L": 10000000,
            "25L+": 20000000
        }.get(income, 5000000)

        if sa_total and sa_total < income_target:
            gap = clamp(0, 25, (1 - sa_total / income_target) * 25)
            score += gap
            reasons.append(f"Underinsured vs income band ({income or 'unknown'}) (+{gap:.0f})")

    inv = clamp(0, 20, (1 - (client.engagement_score / 100.0)) * 20.0)
    score += inv
    reasons.append(f"Engagement risk (+{inv:.0f})")

    score = clamp(0, 100, score)
    bucket = "red" if score >= 67 else ("yellow" if score >= 34 else "green")
    return {"score": score, "bucket": bucket, "reasons": reasons}


def prob_factor_from_followup(score: float) -> float:
    if score >= 70:
        return 0.85
    if score >= 40:
        return 0.55
    return 0.25


def revenue_forecast_3m(db: Session, tenant_id: str) -> Dict[str, Any]:
    today = date.today()
    months = []
    premium_out = []
    comm_out = []

    clients = db.query(Client).filter(Client.tenant_id == tenant_id).all()
    fu_map = {c.id: followup_score_for_client(db, tenant_id, c)["score"] for c in clients}

    policies = db.query(Policy).filter(
        Policy.tenant_id == tenant_id,
        Policy.status == "active"
    ).all()

    for i in range(3):
        month_start = date(today.year + (today.month - 1 + i) // 12, ((today.month - 1 + i) % 12) + 1, 1)
        mlabel = month_start.strftime("%b %Y")
        months.append(mlabel)

        expected_premium = 0.0
        expected_comm = 0.0

        for p in policies:
            freq = (p.premium_frequency or "yearly").lower()
            include = False
            if freq == "monthly":
                include = True
            elif freq == "quarterly":
                if p.start_date:
                    include = ((month_start.month - p.start_date.month) % 3 == 0)
                else:
                    include = (i % 3 == 0)
            else:
                if p.start_date:
                    include = (p.start_date.month == month_start.month)
                else:
                    include = (month_start.month == 1)

            if not include:
                continue

            fu = fu_map.get(p.client_id, 50.0)
            prob = prob_factor_from_followup(fu)

            premium = float(p.premium_amount or 0.0)
            expected_premium += premium * prob
            expected_comm += premium * float(p.commission_rate or 0.0) * prob

        premium_out.append(round(expected_premium, 2))
        comm_out.append(round(expected_comm, 2))

    return {"months": months, "expected_premium": premium_out, "expected_commission": comm_out}


# ==========================
# FastAPI app
# ==========================
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "name": APP_NAME, "ts": now_utc().isoformat()}


@app.post("/auth/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user.id, "tenant_id": user.tenant_id, "role": user.role})
    return TokenOut(access_token=token)


@app.get("/me", response_model=MeOut)
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=500, detail="Tenant missing")
    return MeOut(
        id=user.id,
        role=user.role,
        name=user.name,
        email=user.email,
        tenant=TenantOut(id=tenant.id, name=tenant.name, country=tenant.country, timezone=tenant.timezone),
    )


@app.post("/dev/seed")
def dev_seed(db: Session = Depends(get_db)):
    try:
        admin_email = "admin@gouravnxmx.demo"

        tenant = db.query(Tenant).filter(Tenant.name == "GouravNxMx Demo").first()
        if not tenant:
            tenant = Tenant(name="GouravNxMx Demo", country="IN", timezone="Asia/Kolkata")
            db.add(tenant)
            db.commit()
            db.refresh(tenant)

        user = db.query(User).filter(User.email == admin_email).first()
        if not user:
            user = User(
                tenant_id=tenant.id,
                role="admin",
                name="Demo Admin",
                email=admin_email,
                password_hash=hash_password("admin123"),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # Seed clients/policies/interactions if empty
        existing = db.query(Client).filter(Client.tenant_id == tenant.id).count()
        if existing == 0:
            # (keep your existing seed data block here exactly as it is)
            pass

        return {"ok": True, "tenant": {"name": tenant.name, "id": tenant.id},
                "login": {"email": admin_email, "password": "admin123"}}

    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB integrity error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Seed failed: {type(e).__name__}: {str(e)}")


@app.get("/clients", response_model=List[ClientOut])
def list_clients(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Client).filter(Client.tenant_id == user.tenant_id).order_by(Client.created_at.desc()).all()
    return [ClientOut(
        id=r.id, name=r.name, phone=r.phone, email=r.email, age=r.age, income_band=r.income_band,
        city=r.city, language_pref=r.language_pref, engagement_score=r.engagement_score,
        risk_score=r.risk_score, created_at=r.created_at
    ) for r in rows]


@app.post("/clients", response_model=ClientOut)
def create_client(payload: ClientIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = Client(
        tenant_id=user.tenant_id,
        name=payload.name,
        phone=payload.phone,
        email=(payload.email.lower() if payload.email else None),
        age=payload.age,
        income_band=payload.income_band,
        city=payload.city,
        language_pref=payload.language_pref,
        engagement_score=50.0,
        risk_score=50.0,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return ClientOut(
        id=row.id, name=row.name, phone=row.phone, email=row.email, age=row.age, income_band=row.income_band,
        city=row.city, language_pref=row.language_pref, engagement_score=row.engagement_score,
        risk_score=row.risk_score, created_at=row.created_at
    )


@app.get("/clients/{client_id}", response_model=ClientOut)
def get_client(client_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.query(Client).filter(Client.tenant_id == user.tenant_id, Client.id == client_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Client not found")
    return ClientOut(
        id=row.id, name=row.name, phone=row.phone, email=row.email, age=row.age, income_band=row.income_band,
        city=row.city, language_pref=row.language_pref, engagement_score=row.engagement_score,
        risk_score=row.risk_score, created_at=row.created_at
    )


@app.delete("/clients/{client_id}")
def delete_client(client_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.query(Client).filter(Client.tenant_id == user.tenant_id, Client.id == client_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Client not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


@app.get("/clients/{client_id}/policies", response_model=List[PolicyOut])
def list_policies(client_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.tenant_id == user.tenant_id, Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    rows = db.query(Policy).filter(Policy.tenant_id == user.tenant_id, Policy.client_id == client_id).order_by(Policy.created_at.desc()).all()
    return [PolicyOut(
        id=r.id, client_id=r.client_id, carrier=r.carrier, policy_type=r.policy_type, premium_amount=r.premium_amount,
        premium_frequency=r.premium_frequency, start_date=r.start_date, maturity_date=r.maturity_date,
        sum_assured=r.sum_assured, commission_rate=r.commission_rate, status=r.status, created_at=r.created_at
    ) for r in rows]


@app.post("/policies", response_model=PolicyOut)
def create_policy(payload: PolicyIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.tenant_id == user.tenant_id, Client.id == payload.client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    row = Policy(
        tenant_id=user.tenant_id,
        client_id=payload.client_id,
        carrier=payload.carrier,
        policy_type=payload.policy_type,
        premium_amount=payload.premium_amount,
        premium_frequency=payload.premium_frequency,
        start_date=payload.start_date,
        maturity_date=payload.maturity_date,
        sum_assured=payload.sum_assured,
        commission_rate=payload.commission_rate,
        status=payload.status,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return PolicyOut(
        id=row.id, client_id=row.client_id, carrier=row.carrier, policy_type=row.policy_type, premium_amount=row.premium_amount,
        premium_frequency=row.premium_frequency, start_date=row.start_date, maturity_date=row.maturity_date,
        sum_assured=row.sum_assured, commission_rate=row.commission_rate, status=row.status, created_at=row.created_at
    )


@app.delete("/policies/{policy_id}")
def delete_policy(policy_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.query(Policy).filter(Policy.tenant_id == user.tenant_id, Policy.id == policy_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Policy not found")
    db.delete(row)
    db.commit()
    return {"ok": True}


@app.post("/interactions")
def create_interaction(payload: InteractionIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.tenant_id == user.tenant_id, Client.id == payload.client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    row = Interaction(
        tenant_id=user.tenant_id,
        client_id=payload.client_id,
        user_id=user.id,
        channel=payload.channel,
        occurred_at=payload.occurred_at or now_utc(),
        notes=payload.notes,
        sentiment_score=payload.sentiment_score,
        objection_tags_json=json.dumps(payload.objection_tags or []),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"ok": True, "id": row.id}


@app.post("/insights/recompute")
def recompute_scores(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.tenant_id == user.tenant_id).all()
    updated = 0
    for c in clients:
        r = risk_score_for_client(db, user.tenant_id, c)
        c.risk_score = float(r["score"])
        updated += 1
    db.commit()
    return {"ok": True, "updated_clients": updated}


@app.get("/insights/top-followups", response_model=List[FollowupRow])
def top_followups(limit: int = 20, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.tenant_id == user.tenant_id).all()
    rows: List[FollowupRow] = []
    for c in clients:
        f = followup_score_for_client(db, user.tenant_id, c)
        rows.append(FollowupRow(
            client_id=c.id,
            client_name=c.name,
            followup_score=float(f["score"]),
            bucket=f["bucket"],
            reasons=f["reasons"][:4],
        ))
    rows.sort(key=lambda x: x.followup_score, reverse=True)
    return rows[:limit]


@app.get("/insights/portfolio-heatmap", response_model=List[HeatmapRow])
def portfolio_heatmap(limit: int = 200, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.tenant_id == user.tenant_id).all()
    out: List[HeatmapRow] = []
    for c in clients:
        r = risk_score_for_client(db, user.tenant_id, c)
        out.append(HeatmapRow(
            client_id=c.id,
            client_name=c.name,
            risk_score=float(r["score"]),
            bucket=r["bucket"],
            reasons=r["reasons"][:4],
        ))
    out.sort(key=lambda x: x.risk_score, reverse=True)
    return out[:limit]


@app.get("/insights/revenue-forecast", response_model=ForecastOut)
def revenue_forecast(months: int = 3, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if months != 3:
        raise HTTPException(status_code=400, detail="Only months=3 supported in v1")
    f = revenue_forecast_3m(db, user.tenant_id)
    return ForecastOut(months=f["months"], expected_commission=f["expected_commission"], expected_premium=f["expected_premium"])
