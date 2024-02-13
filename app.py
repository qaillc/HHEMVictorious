import streamlit as st
import requests
import json
import os
import pandas as pd
from sentence_transformers import CrossEncoder
import numpy as np
import re

from textwrap import dedent
import google.generativeai as genai


# Tool import
from crewai.tools.gemini_tools import GeminiSearchTools
from crewai.tools.mixtral_tools import MixtralSearchTools
from crewai.tools.zephyr_tools import ZephyrSearchTools
from crewai.tools.phi2_tools import Phi2SearchTools


# Google Langchain
from langchain_google_genai import GoogleGenerativeAI

#Crew imports
from crewai import Agent, Task, Crew, Process

# Retrieve API Key from Environment Variable
GOOGLE_AI_STUDIO = os.environ.get('GOOGLE_API_KEY')

# Ensure the API key is available
if not GOOGLE_AI_STUDIO:
    raise ValueError("API key not found. Please set the GOOGLE_AI_STUDIO2 environment variable.")

# Set gemini_llm
gemini_llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_AI_STUDIO)

# CrewAI +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def crewai_process_gemini(research_topic):
    # Define your agents with roles and goals
    GeminiAgent = Agent(
        role='Summary Evaluator',
        goal='To learn how to manage her anxiety in social situations through group therapy.',
        backstory="""Skilled in running query evaluation""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                GeminiSearchTools.gemini_search
                   
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a one paragraph summary of the {research_topic}""",
        agent=GeminiAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[GeminiAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result



def crewai_process_mixtral_crazy(research_topic):
    # Define your agents with roles and goals
    MixtralCrazyAgent = Agent(
        role='Summary Evaluator',
        goal='Evaluate the summary using the HHEM-Victara Tuner',
        backstory="""Skilled in running query evaluation""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_crazy      
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a one paragraph summary of the {research_topic}""",
        agent=MixtralCrazyAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[MixtralCrazyAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_mixtral_normal(research_topic):
    # Define your agents with roles and goals
    MixtralNormalAgent = Agent(
        role='Summary Evaluator',
        goal='Evaluate the summary using the HHEM-Victara Tuner',
        backstory="""Skilled in running query evaluation""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                MixtralSearchTools.mixtral_normal      
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a one paragraph summary of the {research_topic}""",
        agent=MixtralNormalAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[MixtralNormalAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_zephyr_normal(research_topic):
    # Define your agents with roles and goals
    ZephrNormalAgent = Agent(
        role='Summary Evaluator',
        goal='Evaluate the summary using the HHEM-Victara Tuner',
        backstory="""Skilled in running query evaluation""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                ZephyrSearchTools.zephyr_normal     
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a one paragraph summary of the {research_topic}""",
        agent=ZephrNormalAgent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[ZephrNormalAgent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result


def crewai_process_phi2(research_topic):
    # Define your agents with roles and goals
    Phi2Agent = Agent(
        role='Emily Mental Patient Graphic Designer Anxiety',
        goal='Evaluate the summary using the HHEM-Victara Tuner',
        backstory="""Skilled in running query evaluation""",
        verbose=True,
        allow_delegation=False,
        llm = gemini_llm,
        tools=[
                Phi2SearchTools.phi2_search     
      ]

    )


    # Create tasks for your agents
    task1 = Task(
        description=f"""Create a one paragraph summary of the {research_topic}""",
        agent=Phi2Agent
    )

    # Instantiate your crew with a sequential process
    crew = Crew(
        agents=[Phi2Agent],
        tasks=[task1],
        verbose=2,
        process=Process.sequential
    )

    # Get your crew to work!
    result = crew.kickoff()
    
    return result





# Initialize the HHEM model +++++++++++++++++++++++++++++++++++++++++++++++
model = CrossEncoder('vectara/hallucination_evaluation_model')

# Function to compute HHEM scores
def compute_hhem_scores(texts, summary):
    pairs = [[text, summary] for text in texts]
    scores = model.predict(pairs)
    return scores

# Define the Vectara query function
def vectara_query(query: str, config: dict):
    corpus_key = [{
        "customerId": config["customer_id"],
        "corpusId": config["corpus_id"],
        "lexicalInterpolationConfig": {"lambda": config.get("lambda_val", 0.5)},
    }]
    data = {
        "query": [{
            "query": query,
            "start": 0,
            "numResults": config.get("top_k", 10),
            "contextConfig": {
                "sentencesBefore": 2,
                "sentencesAfter": 2,
            },
            "corpusKey": corpus_key,
            "summary": [{
                "responseLang": "eng",
                "maxSummarizedResults": 5,
            }]
        }]
    }

    headers = {
        "x-api-key": config["api_key"],
        "customer-id": config["customer_id"],
        "Content-Type": "application/json",
    }
    response = requests.post(
        headers=headers,
        url="https://api.vectara.io/v1/query",
        data=json.dumps(data),
    )
    if response.status_code != 200:
        st.error(f"Query failed (code {response.status_code}, reason {response.reason}, details {response.text})")
        return [], ""

    result = response.json()
    responses = result["responseSet"][0]["response"]
    summary = result["responseSet"][0]["summary"][0]["text"]

    res = [[r['text'], r['score']] for r in responses]
    return res, summary


# Create the main app with three tabs
tab1, tab2, tab3, tab4 = st.tabs(["Synthetic Data", "Data Query", "HHEM-Victara Query Tuner", "Model Evaluation"])

with tab1:

    st.header("Synthetic Data")
    st.link_button("Create Synthetic Medical Data", "https://chat.openai.com/g/g-XyHciw52w-synthetic-clinical-data")

    text1 = """You are an experienced medical doctor with extensive experience in creating medical records, when clicking "Create Data" create synthetic data with the following Elements similar to the given Example following the Condition. If a number is entered product that many synthetic cases varying the details, if not just produce one case.

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
    
    """
    
    
    st.text_area('Algorithm:', text1, height=400)
 

with tab2:
    st.header("Data Query")
    st.link_button("Query & Summarize Data", "https://chat.openai.com/g/g-9tWqg4gRY-explore-summarize-medical-data")

    text2 = """When clicking on "Search Data", request the Case Number.  Search knowledge  for SearchMyData where XXXX is the number given and give the Elements under SearchMyData .  DO NOT SEARCH THE WEB.

    Elements: Case Number: XXXX, Chief Complaint (CC), History of Present Illness (HPI), Past Medical History (PMH), Medication History, Social History (SH), Family History (FH), Review of Systems (ROS), Physical Examination (PE), Diagnostic Test Results, Assessment and Plan, Problem List
    
    SearchMyData: "Case Number": XXXX, "Chief Complaint (CC)":
    
    """
    
    st.text_area('Algorithm:', text2, height=250)
   
with tab3:
    
    st.header("HHEM-Victara Query Tuner")
    
    # User inputs
    query = st.text_area("Enter your text for query tuning", "", height=100)
    lambda_val = st.slider("Lambda Value", min_value=0.0, max_value=1.0, value=0.5)
    top_k = st.number_input("Top K Results", min_value=1, max_value=50, value=10)
    
    
    if st.button("Query Vectara"):
        config = {
    
            "api_key": os.environ.get("VECTARA_API_KEY", ""),
            "customer_id": os.environ.get("VECTARA_CUSTOMER_ID", ""),
            "corpus_id": os.environ.get("VECTARA_CORPUS_ID", ""),      
    
            "lambda_val": lambda_val,
            "top_k": top_k,
        }
    
        results, summary = vectara_query(query, config)
    
        if results:
            st.subheader("Summary")
            st.write(summary)
            
            st.subheader("Top Results")
            
            # Extract texts from results
            texts = [r[0] for r in results[:5]]
            
            # Compute HHEM scores
            scores = compute_hhem_scores(texts, summary)
            
            # Prepare and display the dataframe
            df = pd.DataFrame({'Fact': texts, 'HHEM Score': scores})
            st.dataframe(df)
        else:
            st.write("No results found.")

with tab4:
    
    st.header("Model Evaluation")

    # User input for the research topic
    research_topic = st.text_area('Enter your research topic:', '', height=100)


    # Selection box for the function to execute
    process_selection = st.selectbox(
        'Choose the process to run:',
        ('crewai_process_gemini', 'crewai_process_mixtral_crazy', 'crewai_process_mixtral_normal', 'crewai_process_zephyr_normal', 'crewai_process_phi2')
    )

    # Button to execute the chosen function
    if st.button('Run Process'):
        if research_topic:  # Ensure there's a topic provided
            if process_selection == 'crewai_process_gemini':
                result = crewai_process_gemini(research_topic)
            elif process_selection == 'crewai_process_mixtral_crazy':
                result = crewai_process_mixtral_crazy(research_topic)
            elif process_selection == 'crewai_process_mixtral_normal':
                result = crewai_process_mixtral_normal(research_topic)
            elif process_selection == 'crewai_process_zephyr_normal':
                result = crewai_process_zephyr_normal(research_topic)
            elif process_selection == 'crewai_process_phi2':
                result = crewai_process_phi2(research_topic)
            st.write(result)
        else:
            st.warning('Please enter a research topic.')
