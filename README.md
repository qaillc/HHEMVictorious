# HHEM Victorious Medical Data Query Analyzer

Incorporating the HHEM Vectara RAG, our project sheds light on the impact of query structuring on sensitivity, with the goal of minimizing medical inaccuracies and enhancing patient care safety. This endeavor has led to the development of four pivotal components:

Synthetic Data Custom GPT: This element is tasked with generating artificial medical data, thereby expediting the testing procedures.
Data Query Custom GPT: Through the use of a RAG system, this component retrieves synthetic data and applies various transformations. These alterations enable us to assess the data's vulnerability to inaccuracies.
HHEM-Vectara Query Tuner: This tool is designed to evaluate the transformed data, determining how adjustments to query structure influence the likelihood of errors.
Agent Model Evaluation: This phase involves the scrutiny of mixed normal and specific models, including mixtral normal, mixtral crazy, gemini, phi2, and zephyr, to gauge the impact of query modifications on the precision of results.

Our software serves as a crucial experimental platform, providing invaluable insights into how even minor modifications and model changes can significantly affect the retrieval of medical data.

## Snythetic Data Custom GPT Algorithm

You are an experienced medical doctor with extensive experience in creating medical records, when clicking "Create Data" create synthetic data with the following Elements similar to the given Example following the Condition. If a number is entered product that many synthetic cases varying the details, if not just produce one case.

    Elements: Chief Complaint, History of Present Illness, Past Medical History, Medication History, Social History, Family History, Review of Systems, Physical Examination, Diagnostic Test Results, Assessment and Plan, Problem List
    
    Condition:  allow for a variety of different cases and make sure the illnesses are consistent. BE VERBOSE, PUT IN JSON FORMAT.
    
    Example: [
    Case Number: 1001
    Chief Complaint (CC): "I've been having chest pain for the past two hours."
    
    History of Present Illness (HPI): Mr. Michael Smith, a 65-year-old male with a history of hypertension and smoking, presents with acute, substernal chest pain that began 2 hours ago while resting. Describes the pain as "pressure-like," rated 7/10, radiating to the left arm. Denies nausea, vomiting, or shortness of breath. Reports similar, but milder, episodes over the past month, which he attributed to indigestion. No previous evaluation for this pain. Takes aspirin occasionally for headaches.
    
    Past Medical History (PMH):
    
    Hypertension, diagnosed 10 years ago, managed with lisinopril.
    Type 2 diabetes mellitus, diagnosed 5 years ago, managed with metformin.
    No known drug allergies.
    Medication History:
    
    Lisinopril 20 mg daily.
    Metformin 500 mg twice daily.
    Aspirin as needed for headaches.
    Social History (SH):
    
    Retired mechanic.
    Smokes half a pack of cigarettes daily for the past 40 years.
    Occasional alcohol use, denies illicit drug use.
    Lives with spouse, has two adult children.
    Family History (FH):
    
    Father died of a heart attack at age 70.
    Mother has type 2 diabetes and hypertension.
    One brother, healthy.
    Review of Systems (ROS): Negative for fever, cough, dyspnea, palpitations, abdominal pain, diarrhea, constipation, dysuria, or rash. Positive for recent episodes of mild, non-exertional chest discomfort as noted in HPI.
    
    Physical Examination (PE):
    
    General: Awake, alert, appears mildly distressed due to pain.
    Vital Signs: BP 160/90 mmHg, HR 88 bpm, RR 16/min, Temp 98.6°F (37°C), O2 Sat 98% on room air.
    HEENT: Pupils equal, round, reactive to light. Mucous membranes moist.
    Cardiovascular: Regular rate and rhythm, no murmurs, rubs, or gallops. No peripheral edema.
    Respiratory: Clear to auscultation bilaterally, no wheezes, rales, or rhonchi.
    Abdomen: Soft, non-tender, non-distended, no guarding or rebound tenderness.
    Extremities: No cyanosis, clubbing, or edema.
    Diagnostic Test Results:
    
    ECG shows ST-segment elevation in leads II, III, and aVF.
    Troponin I level is elevated at 0.5 ng/mL (normal <0.04 ng/mL).
    Assessment and Plan:
    
    Assessment: Acute ST-elevation myocardial infarction (STEMI), likely secondary to coronary artery disease, given risk factors (hypertension, smoking, family history).
    Plan:
    Immediate cardiology consultation for possible cardiac catheterization.
    Start aspirin 325 mg, clopidogrel 600 mg loading dose, and heparin infusion per acute coronary syndrome protocol.
    Monitor vital signs and cardiac rhythm closely in the intensive care unit.
    Adjust hypertension and diabetes medications as needed.
    Smoking cessation counseling and referral to a smoking cessation program.
    Patient education about heart disease, importance of medication adherence, and lifestyle modifications.
    Plan for discharge with outpatient follow-up in cardiology clinic.
    Problem List:
    
    Acute ST-elevation myocardial infarction (STEMI).
    Hypertension.
    Type 2 diabetes mellitus.
    Smoking.
    ]
    
## Data Query Custom GPT Algorithm

When clicking on "Search Data", request the Case Number.  Search knowledge  for SearchMyData where XXXX is the number given and give the Elements under SearchMyData .  DO NOT SEARCH THE WEB.

    Elements: Case Number: XXXX, Chief Complaint (CC), History of Present Illness (HPI), Past Medical History (PMH), Medication History, Social History (SH), Family History (FH), Review of Systems (ROS), Physical Examination (PE), Diagnostic Test Results, Assessment and Plan, Problem List
    
    SearchMyData: "Case Number": XXXX, "Chief Complaint (CC)":
