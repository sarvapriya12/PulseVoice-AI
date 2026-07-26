from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
from datetime import datetime
import uuid

from core.models import Appointment, WaitlistEntry, Patient, Provider, PatientMessage

class AppointmentRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_patient_by_phone(self, phone: str):
        result = await self.db.execute(select(Patient).where(Patient.phone_number == phone))
        return result.scalars().first()

    async def create_patient(self, name: str, phone: str, age: int):
        parts = name.split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""
        patient = Patient(first_name=first_name, last_name=last_name, phone_number=phone, age=age)
        self.db.add(patient)
        await self.db.commit()
        await self.db.refresh(patient)
        return patient

    async def get_default_provider(self):
        result = await self.db.execute(select(Provider))
        return result.scalars().first()

    async def get_appointments_on_date(self, date_obj: datetime.date):
        """
        Get all appointments on a specific date.
        
        Args:
            date_obj (datetime.date): The date to check.
            
        Returns:
            list: A list of Appointment objects.
        """
        start = datetime.combine(date_obj, datetime.min.time())
        end = datetime.combine(date_obj, datetime.max.time())
        result = await self.db.execute(
            select(Appointment).where(and_(Appointment.appointment_time >= start, Appointment.appointment_time <= end))
        )
        return result.scalars().all()

    async def get_appointments_by_phone_and_date(self, phone: str, date_obj: datetime.date):
        start = datetime.combine(date_obj, datetime.min.time())
        end = datetime.combine(date_obj, datetime.max.time())
        result = await self.db.execute(
            select(Appointment)
            .join(Patient, Appointment.patient_id == Patient.id)
            .where(
                and_(
                    Patient.phone_number == phone,
                    Appointment.appointment_time >= start,
                    Appointment.appointment_time <= end,
                    Appointment.status != "cancelled"
                )
            )
        )
        return result.scalars().all()

    async def create_appointment(self, patient_id: uuid.UUID, provider_id: uuid.UUID, appt_time: datetime):
        from sqlalchemy.exc import IntegrityError
        appt = Appointment(patient_id=patient_id, provider_id=provider_id, appointment_time=appt_time, status="scheduled")
        self.db.add(appt)
        try:
            await self.db.commit()
            await self.db.refresh(appt)
            return appt
        except IntegrityError:
            await self.db.rollback()
            raise ValueError(f"Sorry, {appt_time.strftime('%I:%M %p')} was just booked by someone else.")

    async def get_appointment(self, appt_id: str):
        try:
            appt_uuid = uuid.UUID(appt_id)
        except ValueError:
            return None
        result = await self.db.execute(select(Appointment).where(Appointment.id == appt_uuid))
        return result.scalars().first()

    async def update_appointment(self, appt: Appointment):
        await self.db.commit()
        await self.db.refresh(appt)
        return appt

    async def add_to_waitlist(self, patient_name: str, patient_phone: str, date_obj: datetime.date):
        entry = WaitlistEntry(patient_name=patient_name, patient_phone=patient_phone, requested_date=date_obj)
        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)
        return entry

    async def get_waitlist_on_date(self, date_obj: datetime.date):
        result = await self.db.execute(select(WaitlistEntry).where(WaitlistEntry.requested_date == date_obj))
        return result.scalars().all()

    async def save_message(self, patient_name: str, patient_phone: str, message: str):
        msg = PatientMessage(patient_name=patient_name, patient_phone=patient_phone, message=message)
        self.db.add(msg)
        await self.db.commit()
        await self.db.refresh(msg)
        return msg


class AppointmentService:
    def __init__(self, repo: AppointmentRepository):
        self.repo = repo
        self.working_hours = [9, 10, 11, 13, 14, 15, 16] # 9 AM to 4 PM, skipping 12 PM (lunch)

    async def _get_or_create_patient(self, name: str, phone: str, age: int):
        patient = await self.repo.get_patient_by_phone(phone)
        if not patient:
            patient = await self.repo.create_patient(name, phone, age)
        elif patient.age is None:
            # Update age if it was missing
            patient.age = age
            await self.repo.db.commit()
        return patient

    async def check_availability(self, date_str: str, cache: dict = None) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

        if cache and f"slots_{date_str}" in cache:
            appts = cache[f"slots_{date_str}"]
            print(f"[CACHE HIT] Pre-fetched slots loaded for {date_str}")
        else:
            appts = await self.repo.get_appointments_on_date(date_obj)
        booked_hours = [a.appointment_time.hour for a in appts if a.status != "cancelled"]

        open_slots = []
        for hour in self.working_hours:
            if hour not in booked_hours:
                ampm = "AM" if hour < 12 else "PM"
                display_hour = hour if hour <= 12 else hour - 12
                open_slots.append(f"{display_hour} {ampm}")

        if not open_slots:
            return f"I'm sorry, but there are no open slots on {date_str}. Would you like to be added to the waitlist?"

        return f"The doctor is available at the following times on {date_str}: {', '.join(open_slots)}."

    async def book_appointment(self, patient_name: str, patient_phone: str, patient_age: int, time_str: str, cache: dict = None) -> str:
        """
        Book an appointment for a patient at a specific time.
        
        Args:
            patient_name (str): The name of the patient.
            patient_phone (str): The phone number of the patient.
            patient_age (int): The age of the patient.
            time_str (str): The time of the appointment in 'YYYY-MM-DD HH:MM:SS' format.
            
        Returns:
            str: A message indicating the success or failure of the booking.
        """
        if not patient_name or not patient_name.strip():
            return "Booking failed: You must ask the patient for their name."
        if not patient_phone or not patient_phone.strip():
            return "Booking failed: You must ask the patient for their phone number."
        if not patient_age or int(patient_age) <= 0:
            return "Booking failed: You must ask the patient for their age."

        try:
            appt_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            from datetime import timezone
            appt_time = appt_time.replace(tzinfo=timezone.utc)
        except ValueError:
            return "Invalid date format. Please provide the time in YYYY-MM-DD HH:MM:SS format."

        if appt_time < datetime.now(timezone.utc):
            return "Cannot book an appointment in the past. Please choose a future date and time."
            
        if appt_time.hour not in self.working_hours:
            return "Cannot book appointment outside of working hours (9 AM to 4 PM, skipping 12 PM). Please choose a valid time."

        if cache and f"slots_{appt_time.date().strftime('%Y-%m-%d')}" in cache:
            appts = cache[f"slots_{appt_time.date().strftime('%Y-%m-%d')}"]
        else:
            appts = await self.repo.get_appointments_on_date(appt_time.date())
            
        for a in appts:
            if a.appointment_time.hour == appt_time.hour and a.status != "cancelled":
                return f"Sorry, {appt_time.strftime('%I %p')} is already booked. Please choose another time."

        patient = await self._get_or_create_patient(patient_name, patient_phone, patient_age)
        
        if cache and "provider" in cache:
            provider = cache["provider"]
        else:
            provider = await self.repo.get_default_provider()
        
        if not provider:
            return "Cannot book appointment: no provider is configured in the system. Please contact the clinic."

        try:
            await self.repo.create_appointment(patient.id, provider.id, appt_time)
        except ValueError as e:
            return str(e)
            
        readable_time = appt_time.strftime("%A, %B %d at %I:%M %p")
        return f"Done! Your appointment is booked for {readable_time}. We'll send a confirmation text to {patient_phone}."

    async def cancel_appointment_by_phone(self, patient_phone: str, date_str: str, time_str: str = None) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

        appts = await self.repo.get_appointments_by_phone_and_date(patient_phone, date_obj)

        if not appts:
            return f"No active appointment found on {date_str} for that number."

        if time_str:
            try:
                time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
                # Assuming appt.appointment_time is timezone-aware UTC, but time_obj is naive
                # To compare just the time component, we can use .time() on both. 
                # (But wait, the time stored in DB is UTC, we should check if they match)
            except ValueError:
                return "Invalid time format. Please use HH:MM:SS."
            appts = [a for a in appts if a.appointment_time.time() == time_obj]
            if not appts:
                return f"No active appointment found on {date_str} at {time_str}."

        if len(appts) > 1:
            times = ", ".join(a.appointment_time.strftime("%I:%M %p") for a in appts)
            return f"I found {len(appts)} appointments on that date: {times}. Which time would you like to cancel? Specify time_str as HH:MM:SS."

        appt = appts[0]
        readable_time = appt.appointment_time.strftime("%A, %B %d at %I:%M %p")
        appt.status = "cancelled"
        await self.repo.update_appointment(appt)
        return f"Cancelled — your {readable_time} appointment has been removed."

    async def add_to_waitlist(self, patient_name: str, patient_phone: str, patient_age: int, date_str: str) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format. Please provide the date in YYYY-MM-DD format."

        await self._get_or_create_patient(patient_name, patient_phone, patient_age)
        await self.repo.add_to_waitlist(patient_name, patient_phone, date_obj)
        return f"Successfully added {patient_name} to the waitlist for {date_str}."

    async def check_waitlist(self, date_str: str) -> str:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return "Invalid date format. Please provide the date in YYYY-MM-DD format."

        entries = await self.repo.get_waitlist_on_date(date_obj)
        count = len(entries)
        if count == 0:
            return f"There is currently no one on the waitlist for {date_str}."
        return f"There are {count} patients on the waitlist for {date_str}."

    async def take_message(self, patient_name: str, patient_phone: str, message: str) -> str:
        """
        Save a message from the patient for the doctor to review later.
        """
        await self.repo.save_message(patient_name, patient_phone, message)
        return f"Successfully saved message for Dr. Smith from {patient_name}."
