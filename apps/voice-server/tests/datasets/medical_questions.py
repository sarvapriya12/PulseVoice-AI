"""
Medical Questions Dataset for Voice AI Benchmark & Quality Assessment.
Includes 52 realistic, domain-specific clinical inquiries categorized by:
- medication_dosages
- phonetic_twins (confusable pairs: hypertension/hypotension, etc.)
- clinical_abbreviations
- vital_signs_and_measurements
- complex_clinical_triage
- short_questions
- long_questions
"""

import json
from pathlib import Path
from typing import List, Dict, Any

MEDICAL_QUESTIONS: List[Dict[str, Any]] = [
    # ── Category 1: Medication Dosages & Regimens (10 questions) ──
    {
        "id": "med_001",
        "category": "medication_dosages",
        "text": "The patient was prescribed lisinopril 10 milligrams once daily for essential hypertension.",
        "medical_terms": ["lisinopril", "10", "milligrams", "hypertension"],
        "description": "Standard ACE inhibitor dosage and indication."
    },
    {
        "id": "med_002",
        "category": "medication_dosages",
        "text": "Take metformin 500 milligrams twice a day with morning and evening meals.",
        "medical_terms": ["metformin", "500", "milligrams", "meals"],
        "description": "Antidiabetic medication with meal-time scheduling."
    },
    {
        "id": "med_003",
        "category": "medication_dosages",
        "text": "Administer atorvastatin 40 milligrams orally at bedtime for hyperlipidemia.",
        "medical_terms": ["atorvastatin", "40", "milligrams", "hyperlipidemia"],
        "description": "Statin therapy with bedtime frequency."
    },
    {
        "id": "med_004",
        "category": "medication_dosages",
        "text": "The child requires amoxicillin suspension 250 milligrams per 5 milliliters three times daily.",
        "medical_terms": ["amoxicillin", "suspension", "250", "milligrams", "milliliters"],
        "description": "Pediatric liquid suspension concentration and dosing."
    },
    {
        "id": "med_005",
        "category": "medication_dosages",
        "text": "Start levothyroxine 75 micrograms every morning on an empty stomach thirty minutes before breakfast.",
        "medical_terms": ["levothyroxine", "75", "micrograms", "stomach"],
        "description": "Thyroid hormone replacement with microgram units."
    },
    {
        "id": "med_006",
        "category": "medication_dosages",
        "text": "She takes amlodipine 5 milligrams combined with hydrochlorothiazide 12.5 milligrams daily.",
        "medical_terms": ["amlodipine", "5", "milligrams", "hydrochlorothiazide", "12.5"],
        "description": "Dual antihypertensive combination therapy with decimal dosage."
    },
    {
        "id": "med_007",
        "category": "medication_dosages",
        "text": "Prescribe omeprazole 20 milligrams delayed release capsules once daily before eating.",
        "medical_terms": ["omeprazole", "20", "milligrams", "capsules"],
        "description": "Proton pump inhibitor formulation."
    },
    {
        "id": "med_008",
        "category": "medication_dosages",
        "text": "The physician ordered gabapentin 300 milligrams three times a day for neuropathic pain.",
        "medical_terms": ["gabapentin", "300", "milligrams", "neuropathic"],
        "description": "Anticonvulsant dosing for nerve pain."
    },
    {
        "id": "med_009",
        "category": "medication_dosages",
        "text": "Inject insulin glargine 20 units subcutaneously once every evening.",
        "medical_terms": ["insulin", "glargine", "20", "units", "subcutaneously"],
        "description": "Subcutaneous injectable insulin unit measure."
    },
    {
        "id": "med_010",
        "category": "medication_dosages",
        "text": "Take prednisone 10 milligrams daily tapering down by 2.5 milligrams every three days.",
        "medical_terms": ["prednisone", "10", "milligrams", "tapering", "2.5"],
        "description": "Corticosteroid taper schedule with fractional doses."
    },

    # ── Category 2: Phonetic Twins & Confusable Clinical Terms (10 questions) ──
    {
        "id": "med_011",
        "category": "phonetic_twins",
        "text": "Does the chart document chronic arterial hypertension or acute orthostatic hypotension?",
        "medical_terms": ["hypertension", "orthostatic", "hypotension"],
        "confusable_pair": ("hypertension", "hypotension"),
        "description": "Phonetic contrast between high and low blood pressure."
    },
    {
        "id": "med_012",
        "category": "phonetic_twins",
        "text": "The endocrinologist ruled out hyperthyroidism and confirmed subclinical hypothyroidism.",
        "medical_terms": ["hyperthyroidism", "subclinical", "hypothyroidism"],
        "confusable_pair": ("hyperthyroidism", "hypothyroidism"),
        "description": "Phonetic contrast between hyper and hypo thyroid conditions."
    },
    {
        "id": "med_013",
        "category": "phonetic_twins",
        "text": "Is the patient presenting with esophageal dysphagia or expressive dysphasia?",
        "medical_terms": ["esophageal", "dysphagia", "expressive", "dysphasia"],
        "confusable_pair": ("dysphagia", "dysphasia"),
        "description": "Phonetic contrast between swallowing impairment and speech impairment."
    },
    {
        "id": "med_014",
        "category": "phonetic_twins",
        "text": "Blood tests showed profound hypoglycemia rather than diabetic hyperglycemia.",
        "medical_terms": ["hypoglycemia", "diabetic", "hyperglycemia"],
        "confusable_pair": ("hypoglycemia", "hyperglycemia"),
        "description": "Phonetic contrast between low and high serum glucose."
    },
    {
        "id": "med_015",
        "category": "phonetic_twins",
        "text": "The neurologist noted expressive aphasia without significant cerebellar ataxia.",
        "medical_terms": ["aphasia", "cerebellar", "ataxia"],
        "confusable_pair": ("aphasia", "ataxia"),
        "description": "Phonetic contrast between neurological speech deficit and motor incoordination."
    },
    {
        "id": "med_016",
        "category": "phonetic_twins",
        "text": "The sleep study evaluated obstructive sleep apnea versus nocturnal dyspnea.",
        "medical_terms": ["obstructive", "apnea", "nocturnal", "dyspnea"],
        "confusable_pair": ("apnea", "dyspnea"),
        "description": "Respiratory breathing cessation versus shortness of breath."
    },
    {
        "id": "med_017",
        "category": "phonetic_twins",
        "text": "The CT scan identified Crohn's inflammation in the terminal ileum near the right ilium bone.",
        "medical_terms": ["ileum", "ilium", "inflammation"],
        "confusable_pair": ("ileum", "ilium"),
        "description": "Homophone anatomical contrast: intestinal tract vs pelvic bone."
    },
    {
        "id": "med_018",
        "category": "phonetic_twins",
        "text": "Is the acute cognitive decline caused by progressive dementia or fluctuating delirium?",
        "medical_terms": ["dementia", "fluctuating", "delirium"],
        "confusable_pair": ("dementia", "delirium"),
        "description": "Cognitive diagnostic distinction between chronic dementia and acute delirium."
    },
    {
        "id": "med_019",
        "category": "phonetic_twins",
        "text": "The surgeon examined the patient for facial paresis as opposed to total paralysis.",
        "medical_terms": ["paresis", "paralysis"],
        "confusable_pair": ("paresis", "paralysis"),
        "description": "Partial weakness versus complete loss of muscle function."
    },
    {
        "id": "med_020",
        "category": "phonetic_twins",
        "text": "Dermatology distinguished an erythematous rash from deeper ecchymosis.",
        "medical_terms": ["erythematous", "ecchymosis"],
        "confusable_pair": ("erythematous", "ecchymosis"),
        "description": "Dermatological contrast between superficial erythema and contusion/bruising."
    },

    # ── Category 3: Clinical Abbreviations & Acronyms (8 questions) ──
    {
        "id": "med_021",
        "category": "clinical_abbreviations",
        "text": "The 12-lead ECG confirmed normal sinus rhythm without ST segment elevation.",
        "medical_terms": ["ECG", "sinus", "rhythm", "elevation"],
        "description": "Electrocardiogram clinical acronym test."
    },
    {
        "id": "med_022",
        "category": "clinical_abbreviations",
        "text": "Schedule an urgent contrast MRI of the brain and a follow-up non-contrast CT scan.",
        "medical_terms": ["MRI", "brain", "CT", "scan"],
        "description": "Neuroimaging diagnostic acronyms."
    },
    {
        "id": "med_023",
        "category": "clinical_abbreviations",
        "text": "The lab completed a CBC, comprehensive metabolic panel, and fasting lipid profile.",
        "medical_terms": ["CBC", "metabolic", "lipid", "panel"],
        "description": "Standard blood laboratory panels."
    },
    {
        "id": "med_024",
        "category": "clinical_abbreviations",
        "text": "The patient with severe COPD and chronic GERD requested an inhaler refill.",
        "medical_terms": ["COPD", "chronic", "GERD", "inhaler"],
        "description": "Pulmonary and gastrointestinal chronic disease acronyms."
    },
    {
        "id": "med_025",
        "category": "clinical_abbreviations",
        "text": "Take ondansetron 4 milligrams sublingually PRN for nausea and vomiting.",
        "medical_terms": ["ondansetron", "4", "milligrams", "sublingually", "PRN"],
        "description": "Pro re nata (as-needed) prescription notation."
    },
    {
        "id": "med_026",
        "category": "clinical_abbreviations",
        "text": "The prescription instructs 1 tablet BID for ten days then 1 tablet daily.",
        "medical_terms": ["tablet", "BID", "days"],
        "description": "Bis in die (twice daily) Latin abbreviation."
    },
    {
        "id": "med_027",
        "category": "clinical_abbreviations",
        "text": "Is this nasal decongestant available OTC or does it require a DEA controlled substance prescription?",
        "medical_terms": ["decongestant", "OTC", "DEA", "prescription"],
        "description": "Over-the-counter and regulatory agency abbreviations."
    },
    {
        "id": "med_028",
        "category": "clinical_abbreviations",
        "text": "Venous duplex ultrasound was ordered to rule out acute DVT and subsequent PE.",
        "medical_terms": ["ultrasound", "DVT", "PE"],
        "description": "Deep vein thrombosis and pulmonary embolism vascular abbreviations."
    },

    # ── Category 4: Vital Signs, Lab Values & Measurements (8 questions) ──
    {
        "id": "med_029",
        "category": "vital_signs_and_measurements",
        "text": "His resting blood pressure is 142 over 88 millimeters of mercury with a pulse of 76 beats per minute.",
        "medical_terms": ["blood", "pressure", "142", "88", "mercury", "pulse", "76"],
        "description": "Systolic and diastolic blood pressure reading with heart rate."
    },
    {
        "id": "med_030",
        "category": "vital_signs_and_measurements",
        "text": "The triage nurse recorded an oral temperature of 101.4 degrees Fahrenheit and SpO2 of 96 percent on room air.",
        "medical_terms": ["temperature", "101.4", "Fahrenheit", "SpO2", "96", "percent"],
        "description": "Febrile body temperature and peripheral oxygen saturation."
    },
    {
        "id": "med_031",
        "category": "vital_signs_and_measurements",
        "text": "Her hemoglobin A1c decreased from 8.9 percent to 6.4 percent over six months.",
        "medical_terms": ["hemoglobin", "A1c", "8.9", "6.4", "percent"],
        "description": "Glycated hemoglobin percentage point comparison."
    },
    {
        "id": "med_032",
        "category": "vital_signs_and_measurements",
        "text": "The patient's serum creatinine was 1.2 milligrams per deciliter and eGFR was 68 milliliters per minute.",
        "medical_terms": ["creatinine", "1.2", "deciliter", "eGFR", "68"],
        "description": "Renal function metrics with compound unit designations."
    },
    {
        "id": "med_033",
        "category": "vital_signs_and_measurements",
        "text": "Serum potassium was measured at 3.2 milliequivalents per liter indicating mild hypokalemia.",
        "medical_terms": ["potassium", "3.2", "milliequivalents", "hypokalemia"],
        "description": "Electrolyte measurement in mEq/L."
    },
    {
        "id": "med_034",
        "category": "vital_signs_and_measurements",
        "text": "The white blood cell count was elevated at 14.5 thousand cells per microliter.",
        "medical_terms": ["white", "blood", "cell", "14.5", "microliter"],
        "description": "Hematologic cell concentration."
    },
    {
        "id": "med_035",
        "category": "vital_signs_and_measurements",
        "text": "Current body mass index is 27.3 kilograms per meter squared within the overweight category.",
        "medical_terms": ["body", "mass", "index", "27.3", "kilograms"],
        "description": "BMI calculation metric."
    },
    {
        "id": "med_036",
        "category": "vital_signs_and_measurements",
        "text": "Total cholesterol was 230 milligrams per deciliter with LDL at 145 and HDL at 48.",
        "medical_terms": ["cholesterol", "230", "LDL", "145", "HDL", "48"],
        "description": "Full lipid fraction numeric readouts."
    },

    # ── Category 5: Complex Clinical Triage & Differential Diagnosis (8 questions) ──
    {
        "id": "med_037",
        "category": "complex_clinical_triage",
        "text": "If a patient taking warfarin reports sudden severe epistaxis and dark tarry stools, should they proceed immediately to the emergency department?",
        "medical_terms": ["warfarin", "epistaxis", "stools", "emergency"],
        "description": "Anticoagulant toxicity and gastrointestinal bleed emergency triage."
    },
    {
        "id": "med_038",
        "category": "complex_clinical_triage",
        "text": "A sixty-two year old diabetic male reports retrosternal chest pressure radiating to the left jaw accompanied by diaphoresis.",
        "medical_terms": ["diabetic", "retrosternal", "chest", "radiating", "diaphoresis"],
        "description": "Acute coronary syndrome presentation in diabetic patient."
    },
    {
        "id": "med_039",
        "category": "complex_clinical_triage",
        "text": "Can ciprofloxacin be co-administered with antacids containing aluminum hydroxide without impairing bioavailability?",
        "medical_terms": ["ciprofloxacin", "antacids", "aluminum", "hydroxide", "bioavailability"],
        "description": "Fluoroquinolone cation chelation interaction."
    },
    {
        "id": "med_040",
        "category": "complex_clinical_triage",
        "text": "The asthmatic patient developed acute bronchospasm shortly after receiving a non-selective beta blocker for supraventricular tachycardia.",
        "medical_terms": ["bronchospasm", "beta", "blocker", "tachycardia"],
        "description": "Beta blocker contraindication in reactive airway disease."
    },
    {
        "id": "med_041",
        "category": "complex_clinical_triage",
        "text": "Does an anaphylactic penicillin allergy cross-react with third generation cephalosporins such as ceftriaxone?",
        "medical_terms": ["anaphylactic", "penicillin", "allergy", "cephalosporins", "ceftriaxone"],
        "description": "Beta-lactam immunological cross-reactivity inquiry."
    },
    {
        "id": "med_042",
        "category": "complex_clinical_triage",
        "text": "Post-operative monitoring for signs of surgical site infection, dehiscence, and deep venous thrombosis.",
        "medical_terms": ["surgical", "infection", "dehiscence", "thrombosis"],
        "description": "Post-surgical complication surveillance."
    },
    {
        "id": "med_043",
        "category": "complex_clinical_triage",
        "text": "Is combining an SSRI with tramadol contraindicated due to elevated risk of serotonin syndrome?",
        "medical_terms": ["SSRI", "tramadol", "contraindicated", "serotonin", "syndrome"],
        "description": "Neuropsychiatric drug-drug toxicity alert."
    },
    {
        "id": "med_044",
        "category": "complex_clinical_triage",
        "text": "A patient with stage 3 chronic kidney disease requires dose adjustment of gabapentin based on estimated glomerular filtration rate.",
        "medical_terms": ["kidney", "disease", "gabapentin", "glomerular", "filtration"],
        "description": "Renal clearance dosage adjustment."
    },

    # ── Category 6: Short Operational & Clinical Questions (4 questions) ──
    {
        "id": "med_045",
        "category": "short_questions",
        "text": "Do I need to fast before my fasting blood glucose test tomorrow?",
        "medical_terms": ["fast", "fasting", "glucose", "test"],
        "description": "Brief pre-procedure patient inquiry."
    },
    {
        "id": "med_046",
        "category": "short_questions",
        "text": "Is the clinic open for walk-in vaccinations on Saturday?",
        "medical_terms": ["clinic", "vaccinations", "Saturday"],
        "description": "Operational administrative inquiry."
    },
    {
        "id": "med_047",
        "category": "short_questions",
        "text": "Can I refill my albuterol inhaler online through the patient portal?",
        "medical_terms": ["refill", "albuterol", "inhaler", "portal"],
        "description": "Medication refill logistics."
    },
    {
        "id": "med_048",
        "category": "short_questions",
        "text": "Does Medicare Part B cover annual wellness physical exams?",
        "medical_terms": ["Medicare", "wellness", "physical", "exams"],
        "description": "Insurance coverage verification."
    },

    # ── Category 7: Long Multi-Clause Patient Narratives (4 questions) ──
    {
        "id": "med_049",
        "category": "long_questions",
        "text": "I have been experiencing a persistent productive cough with yellowish sputum for ten days, accompanied by low-grade fever, chills, and mild bilateral pleuritic chest discomfort when taking deep breaths.",
        "medical_terms": ["productive", "cough", "sputum", "fever", "pleuritic", "chest"],
        "description": "Detailed respiratory narrative with multiple clinical descriptors."
    },
    {
        "id": "med_050",
        "category": "long_questions",
        "text": "My seventy-year-old mother was diagnosed with congestive heart failure and atrial fibrillation; she is currently taking metoprolol succinate, apixaban, and furosemide, but today she woke up with noticeable bilateral pitting pedal edema.",
        "medical_terms": ["heart", "failure", "atrial", "fibrillation", "metoprolol", "apixaban", "furosemide", "edema"],
        "description": "Complex geriatric cardiovascular multi-drug narrative with acute decompensation signs."
    },
    {
        "id": "med_051",
        "category": "long_questions",
        "text": "Following a laparoscopic cholecystectomy three days ago, the patient reports progressive right upper quadrant tenderness, mild scleral icterus, nausea without emesis, and inability to tolerate oral solid foods.",
        "medical_terms": ["laparoscopic", "cholecystectomy", "tenderness", "scleral", "icterus", "nausea"],
        "description": "Post-cholecystectomy complication presentation with biliary symptoms."
    },
    {
        "id": "med_052",
        "category": "long_questions",
        "text": "The patient complains of unilateral throbbing hemicranial headache associated with photophobia, phonophobia, visual scintillating scotoma, and nausea that typically responds to sumatriptan 50 milligrams.",
        "medical_terms": ["hemicranial", "headache", "photophobia", "phonophobia", "scotoma", "sumatriptan", "50"],
        "description": "Classic neurological migraine narrative with aura and abortive medication."
    }
]


def export_json(output_path: str = None) -> Path:
    """Exports dataset to JSON format."""
    if output_path is None:
        output_path = Path(__file__).parent / "medical_questions.json"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(MEDICAL_QUESTIONS, f, indent=2)
    return output_path


def get_questions(count: int = None, category: str = None) -> List[Dict[str, Any]]:
    """Filters and returns medical questions."""
    questions = MEDICAL_QUESTIONS
    if category:
        questions = [q for q in questions if q["category"].lower() == category.lower()]
    if count is not None and count > 0:
        questions = questions[:count]
    return questions


if __name__ == "__main__":
    out = export_json()
    print(f"Exported {len(MEDICAL_QUESTIONS)} medical questions to {out}")
