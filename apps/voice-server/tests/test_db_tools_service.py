import pytest
import uuid
from datetime import datetime

from core.models import Patient, Provider
from tools.db_tools import AppointmentRepository, AppointmentService

@pytest.mark.asyncio
async def test_check_availability_open(async_db_session):
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)

    response = await service.check_availability("2025-10-15")
    assert "The doctor is available at" in response
    assert "9 AM" in response

@pytest.mark.asyncio
async def test_check_availability_booked(async_db_session):
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)

    # Create dummy patient and provider
    patient = Patient(first_name="John", last_name="Doe", age=30, phone_number="+1234567890")
    provider = Provider(name="Dr. Smith", specialty="General")
    async_db_session.add_all([patient, provider])
    await async_db_session.commit()

    # Book all slots
    for hour in service.working_hours:
        appt_time = datetime(2025, 10, 15, hour, 0, 0)
        await repo.create_appointment(patient.id, provider.id, appt_time)

    response = await service.check_availability("2025-10-15")
    assert "no open slots" in response

@pytest.mark.asyncio
async def test_book_appointment_success(async_db_session):
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)

    response = await service.book_appointment("Jane Doe", "+1098765432", "2025-10-16 10:00:00")
    assert "Successfully booked" in response

@pytest.mark.asyncio
async def test_book_appointment_invalid_date(async_db_session):
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)

    response = await service.book_appointment("Jane Doe", "+1098765432", "10-16-2025")
    assert "Invalid date format" in response

@pytest.mark.asyncio
async def test_cancel_appointment_non_existent(async_db_session):
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)

    response = await service.cancel_appointment(str(uuid.uuid4()))
    assert "Could not find an appointment with that ID" in response
