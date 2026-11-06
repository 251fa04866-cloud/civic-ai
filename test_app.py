"""
CivicAI – End-to-End Automated Test Suite
Validates all core workflows: Authentication, AI Classification, Priority Scoring,
Duplicate Detection, Database CRUD, Staff Assignment, and Analytics.
"""

import sys
import unittest
from pathlib import Path

# UTF-8 encoding for console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import database
from database import (
    init_db,
    create_user,
    get_user_by_username,
    insert_complaint,
    get_complaint_by_id,
    update_complaint_status,
    assign_complaint_to_staff,
    get_all_staff,
    get_status_history,
    get_analytics_metrics
)
import auth
from auth import hash_password, verify_password, login_user
import ai_model
from ai_model import get_classifier_pipeline
import priority_engine
from priority_engine import calculate_priority_score
import duplicate_detector
from duplicate_detector import DuplicateDetector

class TestCivicAI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.pipeline = get_classifier_pipeline()
        cls.detector = DuplicateDetector()

    def test_01_password_hashing_and_auth(self):
        """Test PBKDF2 salted password hashing and authentication."""
        pwd = "SecureCitizenPassword@2026"
        hashed = hash_password(pwd)
        self.assertNotEqual(pwd, hashed)
        self.assertTrue(verify_password(hashed, pwd))
        self.assertFalse(verify_password(hashed, "WrongPassword"))

        # Verify seeded admin login
        admin, err = login_user("admin", "Admin@123")
        self.assertIsNone(err)
        self.assertIsNotNone(admin)
        self.assertEqual(admin["role"], "admin")

        # Verify seeded citizen login
        citizen, err = login_user("citizen", "Citizen@123")
        self.assertIsNone(err)
        self.assertIsNotNone(citizen)
        self.assertEqual(citizen["role"], "citizen")

    def test_02_ai_classification(self):
        """Test AI complaint category prediction across all 5 categories."""
        test_cases = [
            ("Deep pothole on road", "Car tire got damaged due to a crater-like ditch in the tarmac", "Road Damage"),
            ("Garbage bins overflowing", "Rotten waste and plastic scattered across the street with flies", "Garbage"),
            ("Water main burst", "Drinking water is spraying from the underground pipeline leak", "Water Leakage"),
            ("Drainage blocked", "Black sewage water is backing up from the clogged storm drain", "Blocked Drainage"),
            ("Dark streetlights", "Streetlight lamps on the entire road are not functioning at night", "Broken Streetlights")
        ]

        for title, desc, expected_cat in test_cases:
            res = self.pipeline.predict_one(title, desc)
            self.assertEqual(res["category"], expected_cat)
            self.assertGreater(res["confidence"], 0.40)
            self.assertIn(expected_cat, res["probabilities"])

    def test_03_priority_engine(self):
        """Test rule-based multi-factor priority calculation."""
        # Low Priority scenario
        p_low = calculate_priority_score("Minor", "No Immediate Hazard", "Isolated (Few Households)")
        self.assertLessEqual(p_low["priority_score"], 45)
        self.assertEqual(p_low["priority_level"], "Low")

        # Critical Priority scenario
        p_crit = calculate_priority_score("Severe", "Life-Threatening / Severe Hazard", "Arterial Road / Major Public Transit Corridor")
        self.assertGreaterEqual(p_crit["priority_score"], 85)
        self.assertEqual(p_crit["priority_level"], "Critical")

    def test_04_duplicate_detection(self):
        """Test TF-IDF Cosine similarity duplicate detection."""
        # Query that is identical/highly similar to one of our demo complaints
        dup_check = self.detector.check_for_duplicates(
            new_title="Severe pothole causing vehicle damage near school",
            new_description="Deep 2-foot pothole on the main carriageway where two-wheelers fell.",
            new_location="MG Road, Near Metro Pillar 142, Ward 12"
        )
        self.assertTrue(dup_check["is_duplicate"])
        self.assertGreaterEqual(dup_check["highest_similarity"], 0.65)
        self.assertIsNotNone(dup_check["matched_complaint"])

    def test_05_complaint_lifecycle(self):
        """Test complete complaint submission, assignment, status update, and history."""
        citizen = get_user_by_username("citizen")
        staff_list = get_all_staff()
        self.assertGreater(len(staff_list), 0)
        officer = staff_list[0]

        # 1. Insert Complaint
        payload = {
            "user_id": citizen["id"],
            "citizen_name": citizen["full_name"],
            "citizen_email": citizen["email"],
            "title": "Broken storm drain cover on 4th cross",
            "description": "Concrete slab over gutter broken, pedestrians risk falling in at night.",
            "location": "Subhash Nagar, 4th Cross",
            "category_ai": "Blocked Drainage",
            "category_confirmed": "Blocked Drainage",
            "confidence_score": 0.92,
            "severity_level": "Major",
            "safety_risk": "Moderate Hazard (Injury Risk)",
            "public_impact": "Local Street / Neighborhood",
            "priority_score": 70,
            "priority_level": "High",
            "status": "Submitted",
            "photo_path": None,
            "is_duplicate": 0,
            "duplicate_of_id": None,
            "duplicate_score": 0.0
        }

        cid = insert_complaint(payload)
        self.assertTrue(cid.startswith("CIVIC-"))

        # 2. Retrieve Complaint
        comp = get_complaint_by_id(cid)
        self.assertIsNotNone(comp)
        self.assertEqual(comp["status"], "Submitted")

        # 3. Assign to Staff Officer
        assign_ok, _ = assign_complaint_to_staff(
            cid,
            officer["id"],
            "Admin: System Test",
            "Urgent repair of slab required."
        )
        self.assertTrue(assign_ok)

        comp_after_assign = get_complaint_by_id(cid)
        self.assertEqual(comp_after_assign["status"], "Assigned")
        self.assertEqual(comp_after_assign["assigned_staff_id"], officer["id"])

        # 4. Update Status to Resolved
        update_ok, _ = update_complaint_status(
            cid,
            "Resolved",
            "Officer: " + officer["name"],
            "New reinforced concrete slab installed successfully."
        )
        self.assertTrue(update_ok)

        comp_after_resolve = get_complaint_by_id(cid)
        self.assertEqual(comp_after_resolve["status"], "Resolved")

        # 5. Verify Status History Audit Trail
        history = get_status_history(cid)
        self.assertGreaterEqual(len(history), 3)  # Submitted -> Assigned -> Resolved

    def test_06_analytics_metrics(self):
        """Test executive dashboard metrics computation."""
        metrics = get_analytics_metrics()
        self.assertGreater(metrics["total"], 0)
        self.assertGreaterEqual(metrics["resolved"], 1)

if __name__ == "__main__":
    unittest.main()
