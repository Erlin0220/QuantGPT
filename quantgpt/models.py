"""SQLAlchemy ORM models for QuantGPT."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=True)  # bcrypt, NULL=未设置密码
    nickname = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    subscribe_weekly = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    sessions = relationship("Session", back_populates="user", lazy="selectin")
    tasks = relationship("Task", back_populates="user", lazy="selectin")
    reports = relationship("Report", back_populates="user", lazy="selectin")


class VerificationCode(Base):
    __tablename__ = "verification_codes"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, index=True)
    code = Column(String(6), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    attempts = Column(Integer, default=0, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_verification_codes_email_used", "email", "used"),
    )


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(200), nullable=True)
    market = Column(String(20), default="a_share", nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user = relationship("User", back_populates="sessions")
    tasks = relationship("Task", back_populates="session", lazy="selectin")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(12), primary_key=True)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(Uuid, ForeignKey("sessions.id"), nullable=True, index=True)
    status = Column(String(30), nullable=False, default="pending")
    task_type = Column(String(50), nullable=True, default="backtest")
    parent_task_id = Column(String(12), ForeignKey("tasks.id"), nullable=True)
    params = Column(JSON, nullable=True)
    expression = Column(Text, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user = relationship("User", back_populates="tasks")
    session = relationship("Session", back_populates="tasks")
    reports = relationship("Report", back_populates="task", lazy="selectin")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    task_id = Column(String(12), ForeignKey("tasks.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User", back_populates="reports")
    task = relationship("Task", back_populates="reports")


class SavedFactor(Base):
    __tablename__ = "saved_factors"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    task_id = Column(String(12), ForeignKey("tasks.id"), nullable=True)
    expression = Column(Text, nullable=False)
    name = Column(String(200), nullable=True)       # 用户自定义名称
    note = Column(Text, nullable=True)              # 备注
    tags = Column(JSON, nullable=True)              # 标签列表
    metrics = Column(JSON, nullable=True)           # 快照：report_metrics
    backtest_summary = Column(JSON, nullable=True)  # 快照：backtest_summary
    params = Column(JSON, nullable=True)            # 回测参数
    report_url = Column(String(500), nullable=True)
    market = Column(String(20), default="a_share", nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user = relationship("User")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    description = Column(Text, nullable=False)
    screenshot_path = Column(String(500), nullable=True)
    task_id = Column(String(12), nullable=True)
    user_agent = Column(String(500), nullable=True)
    page_url = Column(String(500), nullable=True)
    webhook_sent = Column(Boolean, default=False, nullable=False)
    resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User")


class SubmittedAlpha(Base):
    __tablename__ = "submitted_alphas"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    alpha_id = Column(String(50), nullable=False, index=True)
    expression = Column(Text, nullable=False)
    expression_normalized = Column(Text, nullable=True)
    region = Column(String(10), nullable=False, default="USA")
    universe = Column(String(20), nullable=False, default="TOP3000")
    delay = Column(Integer, nullable=False, default=1)
    decay = Column(Integer, nullable=False, default=0)
    neutralization = Column(String(30), nullable=False, default="SUBINDUSTRY")
    truncation = Column(Float, nullable=False, default=0.08)
    tag = Column(String(100), nullable=True)
    sharpe = Column(Float, nullable=True)
    fitness = Column(Float, nullable=True)
    returns = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="submitted")
    submitted_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User")

    __table_args__ = (
        Index("ix_submitted_alphas_user_expr", "user_id", "expression_normalized"),
    )


class WQResearchCandidate(Base):
    __tablename__ = "wq_research_candidates"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    account = Column(String(20), nullable=False, default="primary", index=True)
    alpha_id = Column(String(50), nullable=False, index=True)
    expression = Column(Text, nullable=False)
    region = Column(String(10), nullable=False, default="USA")
    universe = Column(String(20), nullable=False, default="TOP3000")
    delay = Column(Integer, nullable=False, default=1)
    decay = Column(Integer, nullable=False, default=0)
    neutralization = Column(String(30), nullable=False, default="SUBINDUSTRY")
    truncation = Column(Float, nullable=False, default=0.08)
    sharpe = Column(Float, nullable=True)
    fitness = Column(Float, nullable=True)
    returns = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
    priority_score = Column(Float, nullable=False, default=0.0)
    family = Column(String(50), nullable=True)
    hypothesis = Column(Text, nullable=True)
    parent_expression = Column(Text, nullable=True)
    generation = Column(Integer, nullable=False, default=0)
    mutation_type = Column(String(50), nullable=True)
    structure_signature = Column(Text, nullable=True)
    data_fields = Column(JSON, nullable=True)
    dataset_id = Column(String(100), nullable=True)
    validation_status = Column(String(30), nullable=False, default="research_pass")
    robustness_score = Column(Float, nullable=True)
    novelty_score = Column(Float, nullable=True)
    validation_details = Column(JSON, nullable=True)
    tag = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="queued")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_candidates_account_alpha", "account", "alpha_id", unique=True),
        Index("ix_wq_candidates_account_status_priority", "account", "status", "priority_score"),
    )


class WQResearchTrial(Base):
    __tablename__ = "wq_research_trials"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    account = Column(String(20), nullable=False, default="primary", index=True)
    alpha_id = Column(String(50), nullable=True, index=True)
    expression = Column(Text, nullable=False)
    expression_normalized = Column(Text, nullable=False)
    family = Column(String(50), nullable=False, default="unknown", index=True)
    hypothesis = Column(Text, nullable=True)
    parent_expression = Column(Text, nullable=True)
    generation = Column(Integer, nullable=False, default=0)
    mutation_type = Column(String(50), nullable=True)
    status = Column(String(30), nullable=False, default="researched", index=True)
    sharpe = Column(Float, nullable=True)
    fitness = Column(Float, nullable=True)
    returns = Column(Float, nullable=True)
    turnover = Column(Float, nullable=True)
    self_correlation_failed = Column(Boolean, nullable=False, default=False)
    mutation_targets = Column(JSON, nullable=True)
    settings = Column(JSON, nullable=True)
    data_fields = Column(JSON, nullable=True)
    dataset_id = Column(String(100), nullable=True)
    tag = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_trials_account_family", "account", "family"),
        Index("ix_wq_trials_account_expr", "account", "expression_normalized"),
        Index("ix_wq_trials_account_status", "account", "status"),
    )


class WQSubmissionAttempt(Base):
    __tablename__ = "wq_submission_attempts"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    account = Column(String(20), nullable=False, default="primary", index=True)
    alpha_id = Column(String(50), nullable=False, index=True)
    submission_day = Column(String(10), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="RESERVED")
    score_state = Column(String(20), nullable=False, default="PENDING")
    detail = Column(Text, nullable=True)
    points_at_reservation = Column(Float, nullable=True)
    points_status_at_reservation = Column(String(30), nullable=True)
    attributed_points_share = Column(Float, nullable=True)
    attribution_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_attempts_account_day", "account", "submission_day"),
        Index("ix_wq_attempts_account_alpha_day", "account", "alpha_id", "submission_day", unique=True),
    )


class WQSubmissionState(Base):
    __tablename__ = "wq_submission_states"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    account = Column(String(20), nullable=False, unique=True, index=True)
    daily_budget = Column(Integer, nullable=False, default=2)
    last_observed_points = Column(Float, nullable=True)
    last_points_status = Column(String(30), nullable=True)
    last_settled_points = Column(Float, nullable=True)
    last_settled_delta = Column(Float, nullable=True)
    last_settled_submission_count = Column(Integer, nullable=False, default=0)
    untracked_active_gap = Column(Integer, nullable=False, default=0)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    date = Column(String(10), nullable=False)          # "2026-03-24"
    market = Column(String(20), default="a_share", nullable=False)
    title = Column(String(200), nullable=True)
    content = Column(Text, nullable=True)              # markdown
    metrics = Column(JSON, nullable=True)              # index changes, volume, etc.
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_daily_summaries_date_market", "date", "market", unique=True),
    )


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    key_hash = Column(String(255), nullable=False, unique=True)
    prefix = Column(String(10), nullable=False)
    name = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user = relationship("User")
