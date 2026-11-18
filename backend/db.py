from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120), default="")
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    goal: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    weekly_target: Mapped[int] = mapped_column(Integer, default=3)
    sex: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)           # male | female | unspecified
    birth_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    activity_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class BodyProfileRow(Base):
    __tablename__ = "body_profiles"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)


class BaselineRow(Base):
    __tablename__ = "baselines"
    __table_args__ = (UniqueConstraint("user_id", "exercise_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    exercise_id: Mapped[str] = mapped_column(String(80))
    data: Mapped[dict] = mapped_column(JSON)


class Workout(Base):
    __tablename__ = "workouts"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    items: Mapped[list["WorkoutItem"]] = relationship(cascade="all, delete-orphan", order_by="WorkoutItem.position")


class WorkoutItem(Base):
    __tablename__ = "workout_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    workout_id: Mapped[int] = mapped_column(ForeignKey("workouts.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    exercise_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)   # None => unsupported/custom
    exercise_name: Mapped[str] = mapped_column(String(120))
    sets: Mapped[int] = mapped_column(Integer, default=3)
    reps: Mapped[int] = mapped_column(Integer, default=10)
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    rest_seconds: Mapped[int] = mapped_column(Integer, default=60)


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    workout_id: Mapped[Optional[int]] = mapped_column(ForeignKey("workouts.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(120), default="Workout")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    sets: Mapped[list["SetRecord"]] = relationship(cascade="all, delete-orphan", order_by="SetRecord.id")


class SetRecord(Base):
    __tablename__ = "sets"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("workout_sessions.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    exercise_id: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    exercise_name: Mapped[str] = mapped_column(String(120))
    set_number: Mapped[int] = mapped_column(Integer, default=1)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    partial_reps: Mapped[int] = mapped_column(Integer, default=0)
    weight_kg: Mapped[float] = mapped_column(Float, default=0.0)
    avg_form_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_rom: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_rep_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(10), default="manual")      # ai | manual
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now, index=True)
    rep_records: Mapped[list["RepRecord"]] = relationship(cascade="all, delete-orphan")


class RepRecord(Base):
    __tablename__ = "reps"
    id: Mapped[int] = mapped_column(primary_key=True)
    set_id: Mapped[int] = mapped_column(ForeignKey("sets.id", ondelete="CASCADE"), index=True)
    idx: Mapped[int] = mapped_column(Integer)
    counted: Mapped[bool] = mapped_column(Boolean, default=True)
    peak_progress: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)
    form_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    evidence: Mapped[list] = mapped_column(JSON, default=list)


class UnsupportedRequest(Base):
    __tablename__ = "unsupported_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    request_count: Mapped[int] = mapped_column(Integer, default=1)
    candidate_pattern: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    closest: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="new")   # new|triaged|planned|collecting|shipped|declined
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=now)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=now)
    users: Mapped[list["UnsupportedRequestUser"]] = relationship(cascade="all, delete-orphan")


class UnsupportedRequestUser(Base):
    __tablename__ = "unsupported_request_users"
    request_id: Mapped[int] = mapped_column(ForeignKey("unsupported_requests.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[int] = mapped_column(primary_key=True)     # no FK: survives account deletion as a bare count


class CustomExercise(Base):
    """Exercise definitions added through the admin console (config, not code)."""
    __tablename__ = "custom_exercises"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("kind", "name", "version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(20))            # pose | movement | form | recognizer
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[int] = mapped_column(Integer)
    dataset_version: Mapped[str] = mapped_column(String(80), default="")
    status: Mapped[str] = mapped_column(String(20), default="candidate")   # candidate|production|retired
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    supported_exercises: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[str] = mapped_column(Text, default="")
    artifact_uri: Mapped[str] = mapped_column(String(255), default="")
    lineage: Mapped[dict] = mapped_column(JSON, default=dict)        # code_commit, training_config, pose model, ...
    promotion_record: Mapped[dict] = mapped_column(JSON, default=dict)   # gate evaluation snapshot at promotion time
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class FoodLog(Base):
    __tablename__ = "food_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)          # YYYY-MM-DD (user-local date)
    name: Mapped[str] = mapped_column(String(120))
    food_key: Mapped[Optional[str]] = mapped_column(String(60), nullable=True)
    grams: Mapped[float] = mapped_column(Float)
    kcal: Mapped[float] = mapped_column(Float)
    protein_g: Mapped[float] = mapped_column(Float, default=0.0)
    carbs_g: Mapped[float] = mapped_column(Float, default=0.0)
    fat_g: Mapped[float] = mapped_column(Float, default=0.0)


def migrate(engine) -> list:
    """Additive migration: add any column the models define but the live table lacks (nullable/defaulted only).
    Keeps existing databases working without Alembic; swap for Alembic once schema changes need renames/drops."""
    from sqlalchemy import inspect, text
    insp, added = inspect(engine), []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                if not col.nullable and col.default is None and col.server_default is None:
                    raise RuntimeError(f"cannot auto-add NOT NULL column {table.name}.{col.name} without a default")
                ddl = col.type.compile(engine.dialect)
                conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}'))
                added.append(f"{table.name}.{col.name}")
    return added


def make_engine(url: str):
    if url.startswith("sqlite"):
        kw = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in url:
            kw["poolclass"] = StaticPool
        return create_engine(url, **kw)
    return create_engine(url, pool_pre_ping=True)


def make_sessionmaker(engine):
    Base.metadata.create_all(engine)
    migrate(engine)
    return sessionmaker(engine, expire_on_commit=False)
