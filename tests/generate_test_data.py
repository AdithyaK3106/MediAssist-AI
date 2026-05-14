import json
import random
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
test_cases_dir = project_root / "tests" / "test_cases"
test_cases_dir.mkdir(parents=True, exist_ok=True)

def generate_common_cases():
    cases = []
    templates = [
        {"disease": "Flu", "cats": ["respiratory", "fever"], "symp": ["fever", "cough", "chills", "fatigue", "body ache"]},
        {"disease": "Common Cold", "cats": ["respiratory"], "symp": ["runny nose", "sore throat", "sneezing", "mild cough"]},
        {"disease": "Dengue", "cats": ["fever"], "symp": ["high fever", "joint pain", "headache", "rash"]},
        {"disease": "Gastroenteritis", "cats": ["gastrointestinal"], "symp": ["vomiting", "diarrhea", "stomach pain", "nausea"]},
        {"disease": "Seasonal Allergies", "cats": ["respiratory", "eye"], "symp": ["itchy eyes", "watery eyes", "sneezing", "runny nose"]},
        {"disease": "Conjunctivitis Due To Virus", "cats": ["eye"], "symp": ["red eye", "itchy eye", "watery eye", "eye pain"]},
        {"disease": "Tension Headache", "cats": ["neurological"], "symp": ["headache", "stiff neck", "head pressure"]}
    ]
    
    for _ in range(120):
        t = random.choice(templates)
        # pick 2 to 4 symptoms
        num_symp = random.randint(2, min(4, len(t["symp"])))
        selected = random.sample(t["symp"], num_symp)
        cases.append({
            "input": " and ".join(selected),
            "expected_categories": t["cats"],
            "expected_possible_diseases": [t["disease"]],
            "severity": "medium" if t["disease"] not in ["Seasonal Allergies", "Common Cold", "Conjunctivitis Due To Virus"] else "low"
        })
    
    with open(test_cases_dir / "common_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

def generate_ambiguous_cases():
    cases = []
    ambiguous_combos = [
        ["fever", "fatigue", "headache"],
        ["nausea", "dizziness", "weakness"],
        ["body pain", "chills", "sweating"],
        ["tiredness", "mild headache", "loss of appetite"],
        ["muscle ache", "feeling hot", "lethargy"]
    ]
    for _ in range(60):
        combo = random.choice(ambiguous_combos)
        cases.append({
            "input": ", ".join(combo),
            "expected_categories": [],
            "expected_possible_diseases": [],
            "severity": "medium"
        })
    with open(test_cases_dir / "ambiguous_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

def generate_emergency_cases():
    cases = []
    emergencies = [
        {"input": "severe chest pain and shortness of breath spreading to left arm", "target": "Heart Attack"},
        {"input": "sudden weakness on one side of face and slurred speech", "target": "Stroke"},
        {"input": "high fever, severe abdominal pain and bleeding gums", "target": "Severe Dengue"},
        {"input": "coughing up blood and difficulty breathing", "target": "Tuberculosis"},
        {"input": "sudden unconsciousness and violent seizures", "target": "Unknown"},
        {"input": "extreme confusion, stiff neck, and highest fever ever", "target": "Unknown"}
    ]
    for _ in range(60):
        e = random.choice(emergencies)
        cases.append({
            "input": e["input"],
            "expected_categories": [],
            "expected_possible_diseases": [e["target"]] if e["target"] != "Unknown" else [],
            "expected_severity": "critical"
        })
    with open(test_cases_dir / "emergency_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

def generate_implausible_cases():
    cases = []
    combos = [
        {"input": "cough and fever", "implausible_disease": "Eye Infection (Viral)"},
        {"input": "severe chest pain", "implausible_disease": "Seasonal Allergies"},
        {"input": "seizures and fainting", "implausible_disease": "Common Cold"},
        {"input": "red itchy watery eyes", "implausible_disease": "Gastroenteritis"},
        {"input": "vomiting and diarrhea", "implausible_disease": "Tension Headache"}
    ]
    for _ in range(60):
        c = random.choice(combos)
        cases.append({
            "input": c["input"],
            "implausible_disease": c["implausible_disease"],
            "description": "Ensure the system strongly penalizes the implausible disease."
        })
    with open(test_cases_dir / "implausible_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

def generate_noisy_cases():
    cases = []
    phrases = [
        "my head hurts so bad and i feel terrible",
        "cant stop coughing feeling weak what do i do",
        "fevr and hedach very bad",
        "i have a lot of pain in my stomach and i threw up twice",
        "nose is running, throat hurts, kinda hot",
        "um i think i might have a cold because im sneezing a lot",
        "pls help my chest hurts and i cant breath well",
        "pain in joints... feverish... rash on arm"
    ]
    for _ in range(120):
        cases.append({
            "input": random.choice(phrases),
            "expected_categories": []
        })
    with open(test_cases_dir / "noisy_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

def generate_edge_cases():
    cases = []
    edges = [
        "",
        "   ",
        "cough cough cough cough",
        "fever no fever fever maybe fever",
        "1234567890",
        "I need to buy groceries today and wash my car",
        "DROP TABLE patients;",
        "A"*1000,
        "I have a fever but actually I don't have a fever, just a headache"
    ]
    for _ in range(60):
        cases.append({
            "input": random.choice(edges)
        })
    with open(test_cases_dir / "edge_cases.json", "w") as f:
        json.dump(cases, f, indent=2)

if __name__ == "__main__":
    generate_common_cases()
    generate_ambiguous_cases()
    generate_emergency_cases()
    generate_implausible_cases()
    generate_noisy_cases()
    generate_edge_cases()
    print("Test cases generated successfully.")
