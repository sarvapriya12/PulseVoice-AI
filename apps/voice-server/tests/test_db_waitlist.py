import pytest
import uuid
from datetime import datetime
from tools.db_tools import AppointmentRepository, AppointmentService

@pytest.mark.asyncio
async def test_automated_waitlist_flow(async_db_session):
    '''INTEGRATION TEST: Validates the Phase 2 Automated Waitlist Engine logic.'''
    
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)
    
    datetime(2026, 7, 20).date()
    target_date_str = "2026-07-20"
    
    # 1. Assert Waitlist is empty initially
    waitlist_status = await service.check_waitlist(target_date_str)
    assert "no one on the waitlist" in waitlist_status
    
    # 2. Add John Doe to waitlist
    add_response = await service.add_to_waitlist("John Doe", "+15551234567", target_date_str)
    assert "Successfully added John Doe" in add_response
    
    # 3. Add Jane Smith to waitlist (she is 2nd in queue)
    await service.add_to_waitlist("Jane Smith", "+15559876543", target_date_str)
    
    # 4. Check Waitlist — John Doe MUST be first (FIFO Queue Test)
    waitlist_status_2 = await service.check_waitlist(target_date_str)
    assert "2 patients on the waitlist" in waitlist_status_2

@pytest.mark.asyncio
async def test_appointment_cancellation(async_db_session):
    '''UNIT TEST: Canceling an appointment should trigger a waitlist reminder.'''
    
    repo = AppointmentRepository(async_db_session)
    service = AppointmentService(repo)
    
    # Create fake appointment
    appt_time = datetime(2026, 7, 20, 10, 0, 0)
    appt = await repo.create_appointment(uuid.uuid4(), uuid.uuid4(), appt_time)
    
    # Cancel it
    cancel_response = await service.cancel_appointment(str(appt.id))
    
    assert "Successfully cancelled appointment" in cancel_response
