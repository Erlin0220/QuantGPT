"""SQLAlchemy ORM models for QuantGPT."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)  # bcrypt, NULL=未设置密码
    nickname: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    subscribe_weekly: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list[Session]] = relationship("Session", back_populates="user", lazy="selectin")
    tasks: Mapped[list[Task]] = relationship("Task", back_populates="user", lazy="selectin")
    reports: Mapped[list[Report]] = relationship("Report", back_populates="user", lazy="selectin")


class VerificationCode(Base):
    __tablename__ = "verification_codes"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(6), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (Index("ix_verification_codes_email_used", "email", "used"),)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    market: Mapped[str] = mapped_column(String(20), default="a_share", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="sessions")
    tasks: Mapped[list[Task]] = relationship("Task", back_populates="session", lazy="selectin")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(12), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("sessions.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    task_type: Mapped[str | None] = mapped_column(String(50), nullable=True, default="backtest")
    parent_task_id: Mapped[str | None] = mapped_column(String(12), ForeignKey("tasks.id"), nullable=True)
    params: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="tasks")
    session: Mapped[Session | None] = relationship("Session", back_populates="tasks")
    reports: Mapped[list[Report]] = relationship("Report", back_populates="task", lazy="selectin")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(String(12), ForeignKey("tasks.id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="reports")
    task: Mapped[Task] = relationship("Task", back_populates="reports")


class SavedFactor(Base):
    __tablename__ = "saved_factors"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    task_id: Mapped[str | None] = mapped_column(String(12), ForeignKey("tasks.id"), nullable=True)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)  # 用户自定义名称
    note: Mapped[str | None] = mapped_column(Text, nullable=True)  # 备注
    tags: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # 标签列表
    metrics: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # 快照：report_metrics
    backtest_summary: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # 快照：backtest_summary
    params: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # 回测参数
    report_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    market: Mapped[str] = mapped_column(String(20), default="a_share", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User")


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    screenshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(12), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)
    page_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    webhook_sent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User")


class SubmittedAlpha(Base):
    __tablename__ = "submitted_alphas"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    alpha_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    expression_normalized: Mapped[str | None] = mapped_column(Text, nullable=True)
    region: Mapped[str] = mapped_column(String(10), nullable=False, default="USA")
    universe: Mapped[str] = mapped_column(String(20), nullable=False, default="TOP3000")
    delay: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    decay: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    neutralization: Mapped[str] = mapped_column(String(30), nullable=False, default="SUBINDUSTRY")
    truncation: Mapped[float] = mapped_column(Float, nullable=False, default=0.08)
    tag: Mapped[str | None] = mapped_column(String(100), nullable=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="submitted")
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User")

    __table_args__ = (Index("ix_submitted_alphas_user_expr", "user_id", "expression_normalized"),)


class WQKnowledgeSource(Base):
    __tablename__ = "wq_knowledge_sources"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    published_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    access_scope: Mapped[str | None] = mapped_column(String(50), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_metadata: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class WQKnowledgeCard(Base):
    __tablename__ = "wq_knowledge_cards"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    card_key: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    concept: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    family: Mapped[str] = mapped_column(String(80), nullable=False, default="unknown", index=True)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    mechanism: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    scope: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    evidence: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    source_keys: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    operators: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    expression_templates: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    failure_modes: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    mutation_strategies: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="draft", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_knowledge_cards_family_status", "family", "status"),
        Index("ix_wq_knowledge_cards_status_confidence", "status", "confidence"),
    )


class WQResearchCandidate(Base):
    __tablename__ = "wq_research_candidates"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account: Mapped[str] = mapped_column(String(20), nullable=False, default="primary", index=True)
    alpha_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    region: Mapped[str] = mapped_column(String(10), nullable=False, default="USA")
    universe: Mapped[str] = mapped_column(String(20), nullable=False, default="TOP3000")
    delay: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    decay: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    neutralization: Mapped[str] = mapped_column(String(30), nullable=False, default="SUBINDUSTRY")
    truncation: Mapped[float] = mapped_column(Float, nullable=False, default=0.08)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    priority_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    active_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_tier: Mapped[str | None] = mapped_column(String(1), nullable=True)
    probability_support: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    probability_provenance: Mapped[str | None] = mapped_column(String(200), nullable=True)
    calibration_details: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    family: Mapped[str | None] = mapped_column(String(50), nullable=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mutation_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    structure_signature: Mapped[str | None] = mapped_column(Text, nullable=True)
    data_fields: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dataset_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provenance_state: Mapped[str | None] = mapped_column(String(20), nullable=True)
    provenance_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    validation_status: Mapped[str] = mapped_column(String(30), nullable=False, default="research_pass")
    robustness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    novelty_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    self_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    sc_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    local_correlation: Mapped[float | None] = mapped_column(Float, nullable=True)
    local_correlation_alpha_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    local_correlation_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    local_correlation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    validation_details: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    tag: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lineage_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    parent_lineage_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    operator_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    operators: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    mutation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    planner_strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)
    allocation_cell: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    knowledge_card_ids: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_candidates_account_alpha", "account", "alpha_id", unique=True),
        Index("ix_wq_candidates_account_status_priority", "account", "status", "priority_score"),
        Index("ix_wq_candidates_account_status_confidence", "account", "status", "active_probability"),
    )


class WQResearchTrial(Base):
    __tablename__ = "wq_research_trials"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account: Mapped[str] = mapped_column(String(20), nullable=False, default="primary", index=True)
    alpha_id: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    expression: Mapped[str] = mapped_column(Text, nullable=False)
    expression_normalized: Mapped[str] = mapped_column(Text, nullable=False)
    family: Mapped[str] = mapped_column(String(50), nullable=False, default="unknown", index=True)
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_expression: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    mutation_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="researched", index=True)
    sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    fitness: Mapped[float | None] = mapped_column(Float, nullable=True)
    returns: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    self_correlation_failed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_stage: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    failure_reason: Mapped[str | None] = mapped_column(String(60), nullable=True, index=True)
    failure_reasons: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    failure_evidence: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    mutation_targets: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    settings: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    data_fields: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    dataset_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dataset_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    provenance_state: Mapped[str | None] = mapped_column(String(20), nullable=True)
    provenance_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    lineage_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    parent_lineage_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    operator_pattern: Mapped[str | None] = mapped_column(Text, nullable=True)
    operators: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    mutation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    planner_strategy: Mapped[str | None] = mapped_column(String(100), nullable=True)
    allocation_cell: Mapped[str | None] = mapped_column(String(200), nullable=True)
    source_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    knowledge_card_ids: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    tag: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_trials_account_family", "account", "family"),
        Index("ix_wq_trials_account_expr", "account", "expression_normalized"),
        Index("ix_wq_trials_account_status", "account", "status"),
    )


class WQResearchStageEvent(Base):
    __tablename__ = "wq_research_stage_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account: Mapped[str] = mapped_column(String(20), nullable=False, default="primary", index=True)
    lineage_id: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    parent_lineage_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    source_run_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    stage: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    failure_stage: Mapped[str | None] = mapped_column(String(30), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(60), nullable=True)
    details: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_wq_stage_events_account_stage", "account", "stage", "outcome"),
        Index("ix_wq_stage_events_lineage_stage", "lineage_id", "stage"),
    )


class WQSubmissionAttempt(Base):
    __tablename__ = "wq_submission_attempts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account: Mapped[str] = mapped_column(String(20), nullable=False, default="primary", index=True)
    alpha_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    submission_day: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="RESERVED")
    score_state: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    points_at_reservation: Mapped[float | None] = mapped_column(Float, nullable=True)
    points_status_at_reservation: Mapped[str | None] = mapped_column(String(30), nullable=True)
    attributed_points_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    attribution_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    attribution_details: Mapped[Any | None] = mapped_column(JSON, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    __table_args__ = (
        Index("ix_wq_attempts_account_day", "account", "submission_day"),
        Index("ix_wq_attempts_account_alpha_day", "account", "alpha_id", "submission_day", unique=True),
    )


class WQSubmissionState(Base):
    __tablename__ = "wq_submission_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    account: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    daily_budget: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    last_observed_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_points_status: Mapped[str | None] = mapped_column(String(30), nullable=True)
    last_settled_points: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_settled_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_settled_submission_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    untracked_active_gap: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)


class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    date: Mapped[str] = mapped_column(String(10), nullable=False)  # "2026-03-24"
    market: Mapped[str] = mapped_column(String(20), default="a_share", nullable=False)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)  # markdown
    metrics: Mapped[Any | None] = mapped_column(JSON, nullable=True)  # index changes, volume, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (Index("ix_daily_summaries_date_market", "date", "market", unique=True),)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id"), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(String(10), nullable=False)
    name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped[User] = relationship("User")
