import pytest
from datetime import datetime, timezone

from core.models import Patient, Provider, Appointment, CallLog, WaitlistEntry

@pytest.mark.asyncio
async def test_create_patient_provider_appointment(async_db_session):
    # Patient
    patient = Patient(
        first_name="Alice", 
        last_name="Wonderland", 
        dob=datetime(1995, 8, 15).date(), 
        phone_number="+15550001111"
    )
    # Provider
    provider = Provider(name="Dr. Strange", specialty="Surgery")
    
    async_db_session.add_all([patient, provider])
    await async_db_session.commit()
    
    assert patient.id is not None
    assert provider.id is not None
    
    # Appointment
    appt = Appointment(
        patient_id=patient.id, 
        provider_id=provider.id, 
        appointment_time=datetime.now(timezone.utc)
    )
    async_db_session.add(appt)
    await async_db_session.commit()
    await async_db_session.refresh(appt)
    
    assert appt.id is not None
    assert appt.status == "scheduled"  # Checking default status

@pytest.mark.asyncio
async def test_create_call_log_nullable_patient(async_db_session):
    # CallLog without a patient (nullable patient_id)
    log = CallLog(
        call_sid="CA9876543210fedcba", 
        transcript="Hello, I need an appointment.", 
        summary="Patient requesting appointment."
    )
    async_db_session.add(log)
    await async_db_session.commit()
    await async_db_session.refresh(log)
    
    assert log.id is not None
    assert log.patient_id is None
    assert log.call_sid == "CA9876543210fedcba"

@pytest.mark.asyncio
async def test_create_waitlist_entry(async_db_session):
    entry = WaitlistEntry(
        patient_name="Bob Builder", 
        patient_phone="+14445556666", 
        requested_date=datetime(2025, 12, 1).date()
    )
    async_db_session.add(entry)
    await async_db_session.commit()
    await async_db_session.refresh(entry)
    
    assert entry.id is not None
    assert entry.status == "pending"  # Checking default status
