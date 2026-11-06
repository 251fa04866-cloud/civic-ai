"""
CivicAI – Database Seeding Script
Populates initial Admin, Staff, and demo Citizen accounts, along with realistic pre-populated complaints.
"""

import sys
import datetime

# Configure UTF-8 safe output for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from database import (
    init_db,
    create_user,
    get_user_by_username,
    add_staff,
    get_all_staff,
    insert_complaint,
    get_all_complaints
)
from auth import hash_password
from priority_engine import calculate_priority_score
from train_model import train_and_save_model

def seed_database():
    """Initializes schema and seeds default users, staff, and initial complaints."""
    print("--- [CivicAI] Initializing and Seeding Database ---")
    init_db()

    # 1. Seed Default Administrator
    admin_user = get_user_by_username("admin")
    if not admin_user:
        create_user(
            username="admin",
            email="admin@smartcity.gov.in",
            password_hash=hash_password("Admin@123"),
            full_name="Municipal Admin Officer",
            phone="080-22221111",
            role="admin"
        )
        print("✓ Created Default Admin: 'admin' (Password: 'Admin@123')")
    else:
        print("✓ Admin user already exists.")

    # 2. Seed Default Citizen
    citizen_user = get_user_by_username("citizen")
    if not citizen_user:
        create_user(
            username="citizen",
            email="citizen@example.com",
            password_hash=hash_password("Citizen@123"),
            full_name="Aarav Sharma",
            phone="9876543210",
            role="citizen"
        )
        print("✓ Created Demo Citizen: 'citizen' (Password: 'Citizen@123')")
    else:
        print("✓ Citizen user already exists.")

    # Fetch citizen ID for complaint seeding
    citizen_user = get_user_by_username("citizen")
    citizen_id = citizen_user["id"] if citizen_user else 1

    # 3. Seed Municipal Staff Members
    staff_members = get_all_staff()
    if len(staff_members) == 0:
        default_staff = [
            ("Rajesh Kumar", "Roads & Highways Department", "rajesh.kumar@smartcity.gov.in", "9845012341"),
            ("Sunita Rao", "Sanitation & Solid Waste Management", "sunita.rao@smartcity.gov.in", "9845012342"),
            ("Vikram Patel", "Water Supply & Potable Distribution", "vikram.patel@smartcity.gov.in", "9845012343"),
            ("Meera Deshmukh", "Stormwater & Drainage Division", "meera.deshmukh@smartcity.gov.in", "9845012344"),
            ("Anil Verma", "Street Lighting & Electrical Division", "anil.verma@smartcity.gov.in", "9845012345")
        ]
        for name, dept, email, phone in default_staff:
            add_staff(name, dept, email, phone)
        print(f"✓ Seeded {len(default_staff)} Municipal Field Staff Officers.")
    else:
        print(f"✓ Staff officers already present ({len(staff_members)}).")

    # 4. Train Model if not trained
    pipeline = train_and_save_model()

    # 5. Seed Initial Complaints if none exist
    existing = get_all_complaints()
    if len(existing) == 0:
        print("Seeding initial realistic civic complaints for dashboard metrics...")
        demo_complaints = [
            {
                "title": "Severe pothole causing vehicle damage near school",
                "description": "Deep 2-foot pothole on the main carriageway. Multiple two-wheelers have lost balance and fallen. Urgent repair needed.",
                "location": "MG Road, Near Metro Pillar 142, Ward 12",
                "severity_level": "Severe",
                "safety_risk": "Life-Threatening / Severe Hazard",
                "public_impact": "Arterial Road / Major Public Transit Corridor",
                "status": "In Progress"
            },
            {
                "title": "Overflowing community garbage dumpster outside market",
                "description": "Public waste bin has not been cleared for four days. Rotten food waste is attracting stray animals and creating intense foul odor.",
                "location": "Nehru Layout, Main Market Street, Ward 24",
                "severity_level": "Major",
                "safety_risk": "Moderate Hazard (Injury Risk)",
                "public_impact": "Ward / Commercial Market Area",
                "status": "Verified"
            },
            {
                "title": "High pressure drinking water pipeline burst",
                "description": "The primary potable water line ruptured under the walkway. Clean drinking water is gushing into the drain at high velocity.",
                "location": "Indiranagar 100ft Road, Opposite Public Library",
                "severity_level": "Major",
                "safety_risk": "Moderate Hazard (Injury Risk)",
                "public_impact": "Arterial Road / Major Public Transit Corridor",
                "status": "Assigned"
            },
            {
                "title": "Sewage overflowing from blocked manhole onto road",
                "description": "Manhole is completely blocked with silt, discharging black stinking sewage across the driving lane. Walking pedestrians are stranded.",
                "location": "Lakeview Colony, 2nd Main, Near Drainage Canal",
                "severity_level": "Severe",
                "safety_risk": "Moderate Hazard (Injury Risk)",
                "public_impact": "Local Street / Neighborhood",
                "status": "Submitted"
            },
            {
                "title": "Dark streetlights causing safety hazard on highway stretch",
                "description": "Entire 500-meter section has no working streetlights. Commuters and pedestrians are in total darkness after 7 PM.",
                "location": "Brigade Highway, Near Flyover Down-Ramp",
                "severity_level": "Major",
                "safety_risk": "Moderate Hazard (Injury Risk)",
                "public_impact": "Arterial Road / Major Public Transit Corridor",
                "status": "Under Review"
            },
            {
                "title": "Water pipe leak near residential gate",
                "description": "Continuous freshwater dripping from civic pipeline valve joint, causing water pooling on footpath.",
                "location": "Koramangala 5th Block, Near Community Hall",
                "severity_level": "Minor",
                "safety_risk": "Low Hazard (Caution Advised)",
                "public_impact": "Isolated (Few Households)",
                "status": "Resolved"
            },
            {
                "title": "Loose gravel and broken tarmac on turn",
                "description": "Sharp stones and broken bitumen scattered on the sharp turn. Bikers are slipping while turning.",
                "location": "Gandhi Nagar, 3rd Cross, Behind Govt High School",
                "severity_level": "Moderate",
                "safety_risk": "Moderate Hazard (Injury Risk)",
                "public_impact": "Local Street / Neighborhood",
                "status": "Resolved"
            }
        ]

        for item in demo_complaints:
            pred = pipeline.predict_one(item["title"], item["description"])
            p_calc = calculate_priority_score(
                item["severity_level"],
                item["safety_risk"],
                item["public_impact"]
            )

            complaint_record = {
                "user_id": citizen_id,
                "citizen_name": "Aarav Sharma",
                "citizen_email": "citizen@example.com",
                "title": item["title"],
                "description": item["description"],
                "location": item["location"],
                "category_ai": pred["category"],
                "category_confirmed": pred["category"],
                "confidence_score": pred["confidence"],
                "severity_level": item["severity_level"],
                "safety_risk": item["safety_risk"],
                "public_impact": item["public_impact"],
                "priority_score": p_calc["priority_score"],
                "priority_level": p_calc["priority_level"],
                "status": item["status"],
                "photo_path": None,
                "is_duplicate": 0,
                "duplicate_of_id": None,
                "duplicate_score": 0.0
            }
            insert_complaint(complaint_record)

        print(f"✓ Seeded {len(demo_complaints)} pre-configured demo complaints.")
    else:
        print(f"✓ Complaints already exist ({len(existing)}).")

    print("\n--- Seeding Completed Successfully! ---")

if __name__ == "__main__":
    seed_database()
