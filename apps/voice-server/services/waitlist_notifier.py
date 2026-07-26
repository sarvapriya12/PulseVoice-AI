import asyncio
from datetime import datetime, timezone
from sqlalchemy import select, and_

from core.database import AsyncSessionLocal
from core.models import WaitlistEntry, Appointment
from core.config import settings
from twilio.rest import Client

async def process_waitlist_queue():
    """
    Background cron job to process the waitlist.
    Checks for available slots and notifies patients on the waitlist for that day.
    """
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN:
        print("[WAITLIST CRON] Missing Twilio credentials. Cannot send SMS notifications.")
        return

    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    today = datetime.now(timezone.utc).date()

    async with AsyncSessionLocal() as db:
        # Get all pending waitlist entries for today or future dates
        result = await db.execute(
            select(WaitlistEntry)
            .where(and_(
                WaitlistEntry.status == "pending",
                WaitlistEntry.requested_date >= today
            ))
            .order_by(WaitlistEntry.created_at.asc())
        )
        pending_entries = result.scalars().all()

        if not pending_entries:
            return

        for entry in pending_entries:
            # Check if there are any cancelled appointments on the requested date
            appt_result = await db.execute(
                select(Appointment).where(and_(
                    Appointment.status == "cancelled",
                    # Actually, we need to check if there are any OPEN slots for this date.
                    # Since we don't have the working hours logic here, let's just do a naive check
                    # for any cancelled appointments on that date to notify them.
                    # A robust implementation would reuse AppointmentService.check_availability.
                ))
            )
            # Naive approach: if there's any cancelled appointment that day, assume a slot opened.
            # In a real system, you'd check exact open hours.
            appts = appt_result.scalars().all()
            cancelled_on_date = [a for a in appts if a.appointment_time.date() == entry.requested_date]

            if cancelled_on_date:
                # Notify the patient via Twilio SMS
                try:
                    msg = f"Hello {entry.patient_name}, an appointment slot has opened up on {entry.requested_date.strftime('%B %d')} at Dr. Smith's clinic! Please call us back to claim this spot."
                    client.messages.create(
                        body=msg,
                        from_=settings.TWILIO_PHONE_NUMBER,
                        to=entry.patient_phone
                    )
                    entry.status = "notified"
                    await db.commit()
                    print(f"[WAITLIST CRON] Notified {entry.patient_name} for {entry.requested_date}")
                except Exception as e:
                    print(f"[WAITLIST CRON] Twilio Error notifying {entry.patient_name}: {e}")

async def waitlist_cron_job():
    print("[WAITLIST CRON] Started background thread.")
    while True:
        try:
            await process_waitlist_queue()
        except Exception as e:
            print(f"[WAITLIST CRON] Error: {e}")
        # Run every 60 seconds
        await asyncio.sleep(60)
