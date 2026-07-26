'''SQLAlchemy Database Models (Production EHR Schema).

HIPAA-compliant, normalized schema for a medical scheduling system.

Design decisions:
- UUIDs for all primary keys (prevents enumeration attacks)
- Normalized relational tables: Patient, Provider, Appointment, CallLog
- Timezone-aware datetimes (UTC at rest)
- CallLog.transcript stores PHI-scrubbed text only; linked to patient_id
  so the clinic can reconstruct a caller's full history without embedding
  raw PHI in the log table.
- phone_number is indexed for fast caller-ID lookup at call time.
'''

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Date,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    Integer,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid


class Base(DeclarativeBase):
    """
    Declarative base class for SQLAlchemy ORM models.
    Provides common methods and acts as a base for inheritance for other classes.
    """
    pass


def _utcnow() -> datetime:
    """
    Return the current time as a timezone-aware UTC datetime.
    
    Returns:
        datetime: The current UTC datetime.
    """
    return datetime.now(timezone.utc)


class Patient(Base):
    """
    Stores core demographic data for each patient (PHI).

    phone_number is unique and indexed so an inbound Twilio call can be
    matched to a Patient record in O(log n) time.
    """

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Opaque UUID; prevents sequential enumeration of patient records.",
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    phone_number: Mapped[str] = mapped_column(
        String(20),
        unique=True,
        nullable=False,
        doc="E.164 format recommended (e.g. +12025551234).",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # Relationships
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="patient", cascade="all, delete-orphan"
    )
    call_logs: Mapped[list["CallLog"]] = relationship(
        "CallLog", back_populates="patient"
    )

    # Explicit index (also enforced by unique=True above, but named for DDL clarity)
    __table_args__ = (
        Index("ix_patients_phone_number", "phone_number"),
    )

    def __repr__(self) -> str:
        return f"<Patient id={self.id} name='{self.last_name}, {self.first_name}'>"


class Provider(Base):
    """
    Represents a doctor or clinical staff member who can hold appointments.
    """

    __tablename__ = "providers"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    specialty: Mapped[str] = mapped_column(String(200), nullable=False)

    # Relationships
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="provider"
    )

    def __repr__(self) -> str:
        return f"<Provider id={self.id} name='{self.name}' specialty='{self.specialty}'>"


class Appointment(Base):
    """
    Bridge table linking a Patient to a Provider for a scheduled visit.

    status is a plain String rather than a database ENUM so the schema
    stays portable across PostgreSQL, MySQL, and SQLite. Enforce allowed
    values at the application layer or add a CheckConstraint if needed.

    Valid status values: 'scheduled', 'completed', 'canceled', 'no-show'
    """

    __tablename__ = "appointments"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("providers.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    appointment_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="scheduled",
        doc="One of: 'scheduled', 'completed', 'canceled', 'no-show'.",
    )
    reason_for_visit: Mapped[str] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # Relationships
    patient: Mapped["Patient"] = relationship("Patient", back_populates="appointments")
    provider: Mapped["Provider"] = relationship("Provider", back_populates="appointments")

    __table_args__ = (
        UniqueConstraint("provider_id", "appointment_time", name="uq_provider_appointment_time"),
    )

    def __repr__(self) -> str:
        return (
            f"<Appointment id={self.id} "
            f"patient_id={self.patient_id} "
            f"time={self.appointment_time} "
            f"status='{self.status}'>"
        )


class CallLog(Base):
    """
    Audit trail for inbound / outbound calls handled by Twilio.

    CRITICAL PHI NOTE
    -----------------
    The `transcript` column MUST contain ONLY the PHI-scrubbed version of
    the call transcript. Raw transcripts (with names, DOBs, insurance IDs,
    etc.) must be scrubbed by the NLP pipeline BEFORE being written here.
    The `summary` column follows the same rule.

    patient_id is nullable to handle anonymous or unrecognised callers; the
    clinic can retroactively link a call log to a patient if the identity is
    later confirmed.
    """

    __tablename__ = "call_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("patients.id", ondelete="SET NULL"),
        nullable=True,   # Unknown callers allowed
        index=True,
        doc="NULL when the caller could not be identified.",
    )
    call_sid: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        doc="Twilio CallSid (e.g. CA1234abc…).  Used to correlate with Twilio logs.",
    )
    transcript: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="PHI-SCRUBBED transcript only.  Never store raw PHI here.",
    )
    summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        doc="PHI-SCRUBBED AI-generated summary of the call.",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # Relationships
    patient: Mapped["Patient | None"] = relationship(
        "Patient", back_populates="call_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<CallLog id={self.id} "
            f"call_sid='{self.call_sid}' "
            f"patient_id={self.patient_id}>"
        )


class WaitlistEntry(Base):
    """
    Automated Waitlist queue for Phase 2.
    """

    __tablename__ = "waitlist"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    patient_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    requested_date: Mapped[datetime.date] = mapped_column(Date, nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        doc="One of: 'pending', 'notified', 'claimed', 'expired'."
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return f"<WaitlistEntry id={self.id} name='{self.patient_name}' date={self.requested_date} status='{self.status}'>"


class PatientMessage(Base):
    """
    Messages left by patients for the doctor when the AI cannot answer their medical queries.
    """
    __tablename__ = "patient_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_name: Mapped[str] = mapped_column(String(200), nullable=False)
    patient_phone: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="unread", doc="'unread', 'read', 'archived'")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    def __repr__(self) -> str:
        return f"<PatientMessage id={self.id} name='{self.patient_name}' status='{self.status}'>"