"""
CivicAI – Model Training & Synthetic Dataset Generation Script
Generates a realistic 350-sample civic complaint dataset, trains TF-IDF + Logistic Regression,
evaluates cross-validation performance, and serializes artifacts using Joblib.
Designed with safe, pure-Python and standard-library data handling.
"""

import os
import sys
import csv

# Configure UTF-8 safe output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import random
import math
import numpy as np
import joblib
from pathlib import Path
from config import (
    CATEGORIES,
    SYNTHETIC_DATA_PATH,
    MODEL_PATH,
    VECTORIZER_PATH,
    SEVERITY_LEVELS,
    SAFETY_RISK_LEVELS,
    PUBLIC_IMPACT_LEVELS
)
from ai_model import CivicClassifierPipeline

# Template generators for realistic civic complaints
SAMPLE_LOCATIONS = [
    "MG Road, Near Metro Pillar 142, Ward 12",
    "Outer Ring Road, Near Tech Park Junction, Sector 4",
    "Indiranagar 100ft Road, Opposite Public Library",
    "Gandhi Nagar, 3rd Cross, Behind Govt High School",
    "Station Road, Near Central Bus Stand, Old Town",
    "Nehru Layout, Main Market Street, Ward 24",
    "Rajaji Road, Near Civil Hospital Gate 2",
    "Koramangala 5th Block, Near Community Hall",
    "Subhash Nagar, Cross Road 7, Near Water Tank",
    "Shanti Nagar, Double Road, Near City Post Office",
    "Malleshwaram 8th Cross, Near Market Square",
    "JP Nagar 6th Phase, Near Municipal Playground",
    "Brigade Highway, Near Flyover Down-Ramp",
    "Lakeview Colony, 2nd Main, Near Drainage Canal",
    "Commercial Street, Lane 4, Near Cloth Market"
]

ROAD_DAMAGE_TEMPLATES = [
    ("Huge pothole on road causing accidents", "There is a massive 2-foot deep pothole on the main carriageway. Multiple two-wheeler riders have lost balance and fallen over the past two days."),
    ("Tarmac completely washed away after rain", "The entire top asphalt layer of the road has peeled off leaving loose gravel and sharp stones. Driving cars or riding bikes has become extremely dangerous."),
    ("Broken road surface near busy intersection", "The road is completely cracked with uneven ditches. Vehicles are swerving into oncoming traffic to avoid the bad road condition."),
    ("Dangerous sinkhole appearing in asphalt", "A circular depression in the road surface is collapsing into a sinkhole. It needs immediate municipal cordoning before a vehicle collapses inside."),
    ("Loose gravel and broken stones on flyover ramp", "Gravel is scattered all across the turn. Commuters are skidding when applying brakes on the steep incline."),
    ("Crater-like pothole outside school gate", "A very deep ditch has formed right where children cross the road. Water accumulates inside making it impossible to judge depth."),
    ("Damaged road divider and crumbling bitumen", "The asphalt adjacent to the median has crumbled away, leaving deep trenches along the driving lane.")
]

GARBAGE_TEMPLATES = [
    ("Overflowing community garbage dumpster", "The public waste bin has not been cleared for over four days. Solid waste, household garbage, and plastic bags are spilling across the entire street."),
    ("Illegal open garbage dump attracting stray cattle", "A vacant corner plot is being used as a commercial garbage dump. Rotten food waste is attracting stray dogs, cows, and creating a foul stench."),
    ("Rotten vegetable and fish market waste uncollected", "The municipal cleaning truck skipped the market collection today. Decomposing organic waste is rotting under the sun and swarming with flies."),
    ("Construction debris and concrete rubble on sidewalk", "Contractors have dumped tons of cement rubble and broken tiles on the pedestrian footpath, completely blocking foot traffic."),
    ("Hazardous plastic and bio-medical waste scattered", "Syringes, medicine packaging, and untreated plastic waste have been dumped on the roadside near the clinic."),
    ("Dead animal carcass left on road side", "A stray dog carcass has been lying on the pavement since yesterday emitting a severe odor. Immediate sanitary disposal needed."),
    ("Litter and unemptied dustbins in public park", "All public bins in the children's park are overflowing with wrappers, bottles, and food cartons.")
]

WATER_LEAKAGE_TEMPLATES = [
    ("High pressure main water supply pipeline burst", "The primary potable water pipeline has ruptured under the sidewalk. Millions of liters of clean drinking water are gushing into the storm drain."),
    ("Continuous freshwater leak from municipal valve", "A drinking water valve has been spraying clean water continuously for three days. The road is flooded while nearby houses face zero water pressure."),
    ("Drinking water pipeline damaged during fiber optic digging", "Excavators digging utility trenches broke the municipal water main. Water is spewing 10 feet into the air."),
    ("Sub-surface water pipe leaking and eroding road base", "Water is constantly bubbling up through the road surface from a subterranean crack, creating a boggy mud pit."),
    ("Broken municipal water tap wasting drinking water", "The public tap installed near the bus stop has lost its valve handle and is running non-stop day and night."),
    ("Ruptured distribution line flooding residential basement", "Clean water from the civic line is leaking through basement walls of apartment complexes."),
    ("Leaking water meter manifold on public walkway", "Multiple water connection joints are dripping water onto the public pathway creating slippery algae.")
]

BLOCKED_DRAINAGE_TEMPLATES = [
    ("Sewage overflow from blocked manhole onto street", "The municipal sewer manhole is choked with silt and overflowing black, foul-smelling sewage across the road. Pedestrians cannot walk."),
    ("Stormwater drain clogged with plastic causing flash flood", "The roadside open drain is completely plugged with debris and polythene bags. Even light rain results in ankle-deep gutter water inside houses."),
    ("Stagnant foul drain water breeding mosquitoes", "The open culvert has stagnant black water that hasn't flowed in two weeks. Dense mosquito swarms have led to dengue fears among residents."),
    ("Underground drainage backed up into residential bathrooms", "The municipal trunk sewer is obstructed downstream, causing raw sewage backflow into ground-floor toilets and drains."),
    ("Broken concrete slab fallen inside stormwater canal", "A heavy drain covering slab broke and fell into the channel, trapping garbage and completely blocking water flow."),
    ("Clogged drainage trench spreading terrible odor", "Wastewater from local hotels is backing up because the road culvert is jammed with solid grease and waste."),
    ("Monsoon drain silted up with mud and gravel", "Drainage ditch is filled with construction soil. Stormwater is diverting directly into houses.")
]

BROKEN_STREETLIGHTS_TEMPLATES = [
    ("Dark streetlights causing severe safety risk at night", "None of the streetlights on this 500-meter stretch are working. The area is in pitch darkness, leading to thefts and safety hazards for women."),
    ("Blinking and flickering streetlamp pole", "The LED luminaire on pole #14 has been strobe-flickering for a week, causing extreme eye strain and driver distraction."),
    ("Exposed live electrical wires hanging from lamp post", "The maintenance door at the base of the metal streetlight post is open with exposed 230V live cables within reach of children."),
    ("Streetlight pole knocked down by vehicle collision", "A car hit the electric pole last night. The lamppost is leaning precariously over the road held only by overhead wires."),
    ("Entire neighborhood streetlights blackout", "The master lighting timer or circuit breaker tripped. All lights on 2nd and 3rd cross roads have been off for three consecutive nights."),
    ("Streetlight fixture burned out outside senior citizens park", "The sodium vapor lamp has burned out completely, leaving the walking track completely dark."),
    ("Dim and non-functional solar streetlight", "The municipal solar light battery has died and the fixture does not turn on at night.")
]

CATEGORY_TEMPLATES = {
    "Road Damage": ROAD_DAMAGE_TEMPLATES,
    "Garbage": GARBAGE_TEMPLATES,
    "Water Leakage": WATER_LEAKAGE_TEMPLATES,
    "Blocked Drainage": BLOCKED_DRAINAGE_TEMPLATES,
    "Broken Streetlights": BROKEN_STREETLIGHTS_TEMPLATES
}

def generate_synthetic_dataset(total_samples: int = 350) -> list:
    """
    Generates a balanced dataset of 350 synthetic complaints (70 per category).
    Returns list of dictionaries.
    """
    samples_per_category = total_samples // len(CATEGORIES)
    records = []
    random.seed(42)

    severities = list(SEVERITY_LEVELS.keys())
    safety_risks = list(SAFETY_RISK_LEVELS.keys())
    public_impacts = list(PUBLIC_IMPACT_LEVELS.keys())

    for cat in CATEGORIES:
        templates = CATEGORY_TEMPLATES[cat]
        for i in range(samples_per_category):
            title_base, desc_base = random.choice(templates)
            loc = random.choice(SAMPLE_LOCATIONS)

            variation_prefixes = [
                "Urgent attention required: ",
                "Immediate municipal action needed: ",
                "Citizen report: ",
                "Severe issue observed: ",
                "Complaint regarding ",
                "Report of ",
                ""
            ]
            variation_suffixes = [
                " Needs urgent resolution by the zonal officer.",
                " Requesting immediate inspection by municipal engineers.",
                " Please send a team to fix this as soon as possible.",
                " Photos can be provided if required by department staff.",
                " This issue has been persisting for multiple days.",
                " Local residents are facing significant distress.",
                ""
            ]

            prefix = random.choice(variation_prefixes)
            suffix = random.choice(variation_suffixes)

            title = f"{prefix}{title_base.lower() if prefix else title_base}"
            description = f"{desc_base}{suffix}"

            if cat in ["Road Damage", "Broken Streetlights"]:
                sev = random.choices(severities, weights=[0.2, 0.3, 0.35, 0.15])[0]
                risk = random.choices(safety_risks, weights=[0.1, 0.3, 0.4, 0.2])[0]
            elif cat in ["Water Leakage", "Blocked Drainage"]:
                sev = random.choices(severities, weights=[0.15, 0.35, 0.35, 0.15])[0]
                risk = random.choices(safety_risks, weights=[0.2, 0.35, 0.3, 0.15])[0]
            else:  # Garbage
                sev = random.choices(severities, weights=[0.25, 0.4, 0.25, 0.1])[0]
                risk = random.choices(safety_risks, weights=[0.3, 0.4, 0.2, 0.1])[0]

            impact = random.choice(public_impacts)

            records.append({
                "sample_id": f"SYNTH-{len(records) + 1:04d}",
                "title": title.strip(),
                "description": description.strip(),
                "location": loc,
                "category": cat,
                "severity": sev,
                "safety_risk": risk,
                "public_impact": impact,
                "dataset_label": "Synthetic Benchmark (B.Tech Capstone Demonstration)"
            })

    random.shuffle(records)
    return records

def save_dataset_csv(records: list, file_path=SYNTHETIC_DATA_PATH):
    """Save records to CSV using standard library csv module."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id", "title", "description", "location", "category",
        "severity", "safety_risk", "public_impact", "dataset_label"
    ]
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

def load_dataset_csv(file_path=SYNTHETIC_DATA_PATH) -> list:
    """Load records from CSV using standard library csv module."""
    records = []
    with open(file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

def train_and_save_model():
    """
    Trains the TF-IDF Vectorizer + Logistic Regression classifier on the 350-sample dataset,
    evaluates classification performance, and persists models using joblib.
    """
    print("--- [CivicAI] Commencing Model Training Pipeline ---")

    # 1. Load or Generate Dataset
    if SYNTHETIC_DATA_PATH.exists():
        print(f"Loading existing synthetic dataset from: {SYNTHETIC_DATA_PATH}")
        records = load_dataset_csv(SYNTHETIC_DATA_PATH)
    else:
        print(f"Generating realistic 350-sample civic complaint dataset...")
        records = generate_synthetic_dataset(350)
        save_dataset_csv(records, SYNTHETIC_DATA_PATH)
        print(f"Saved dataset to {SYNTHETIC_DATA_PATH} ({len(records)} samples)")

    # 2. Prepare Text and Target Arrays
    texts = [f"{r['title']} {r['description']}" for r in records]
    labels = [r["category"] for r in records]

    # 3. Train-Test Split (80% train, 20% test)
    n_total = len(texts)
    n_train = int(0.8 * n_total)

    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train_text = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    X_test_text = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    print(f"Dataset Split: {len(X_train_text)} Training Samples | {len(X_test_text)} Validation Samples")

    # 4. Train Pipeline
    pipeline = CivicClassifierPipeline()
    pipeline.train(X_train_text, y_train)

    # 5. Evaluate on Validation Set
    correct = 0
    cat_stats = {cat: {"total": 0, "correct": 0} for cat in CATEGORIES}

    for text, true_label in zip(X_test_text, y_test):
        res = pipeline.predict_one("", text)
        pred_label = res["category"]
        cat_stats[true_label]["total"] += 1
        if pred_label == true_label:
            correct += 1
            cat_stats[true_label]["correct"] += 1

    val_accuracy = (correct / len(X_test_text)) * 100.0
    print(f"\nModel Training Complete! Validation Accuracy: {val_accuracy:.2f}% ({correct}/{len(X_test_text)})")
    print("Category-wise Performance Breakdown:")
    for cat, stat in cat_stats.items():
        acc = (stat["correct"] / stat["total"] * 100.0) if stat["total"] > 0 else 0
        print(f"  • {cat:20s}: {acc:5.1f}% ({stat['correct']}/{stat['total']})")

    # 6. Save Artifacts with Joblib
    pipeline.save(MODEL_PATH, VECTORIZER_PATH)
    print(f"\nSerialized Model Artifacts:")
    print(f"  -> Classifier: {MODEL_PATH}")
    print(f"  -> Vectorizer: {VECTORIZER_PATH}")
    print("\n[NOTE]: Disclaimer: Synthetic dataset test accuracy reflects benchmark performance on structured academic prompts and is intended for capstone demonstration purposes.")

    return pipeline

if __name__ == "__main__":
    train_and_save_model()
