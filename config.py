"""
CivicAI – Configuration and Styling Module
Theme: Modern Smart City Government Portal (Navy Blue, White, Light Blue)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"

# Database Configuration
DB_PATH = DATA_DIR / "civic_complaints.db"
SYNTHETIC_DATA_PATH = DATA_DIR / "synthetic_complaints_350.csv"

# Model Paths
MODEL_PATH = MODELS_DIR / "civic_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"

# App Branding
APP_NAME = "CivicAI"
APP_TAGLINE = "Intelligent Civic Complaint Prioritization & Redressal System"
MUNICIPAL_NAME = "Smart City Municipal Corporation"

# Five Standard Categories
CATEGORIES = [
    "Road Damage",
    "Garbage",
    "Water Leakage",
    "Blocked Drainage",
    "Broken Streetlights"
]

CATEGORY_ICONS = {
    "Road Damage": "🚧",
    "Garbage": "🗑️",
    "Water Leakage": "💧",
    "Blocked Drainage": "🌊",
    "Broken Streetlights": "💡"
}

CATEGORY_DESCRIPTIONS = {
    "Road Damage": "Potholes, broken tarmac, cave-ins, missing manhole covers on roads",
    "Garbage": "Overflowing public bins, unattended waste dumps, littering, bio-waste",
    "Water Leakage": "Burst pipelines, continuous freshwater leaks, leaking municipal valves",
    "Blocked Drainage": "Clogged storm drains, sewage overflow, stagnant wastewater, blocked culverts",
    "Broken Streetlights": "Dark streetlights, malfunctioning fixtures, exposed electric wires on poles"
}

# Complaint Statuses
STATUS_SUBMITTED = "Submitted"
STATUS_UNDER_REVIEW = "Under Review"
STATUS_VERIFIED = "Verified"
STATUS_ASSIGNED = "Assigned"
STATUS_IN_PROGRESS = "In Progress"
STATUS_RESOLVED = "Resolved"
STATUS_REJECTED = "Rejected"

COMPLAINT_STATUSES = [
    STATUS_SUBMITTED,
    STATUS_UNDER_REVIEW,
    STATUS_VERIFIED,
    STATUS_ASSIGNED,
    STATUS_IN_PROGRESS,
    STATUS_RESOLVED,
    STATUS_REJECTED
]

# Priority Levels & Badges
PRIORITY_LOW = "Low"
PRIORITY_MEDIUM = "Medium"
PRIORITY_HIGH = "High"
PRIORITY_CRITICAL = "Critical"

PRIORITY_LEVELS = [
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL
]

PRIORITY_COLORS = {
    PRIORITY_LOW: "#28A745",       # Green
    PRIORITY_MEDIUM: "#E67E22",    # Amber/Orange
    PRIORITY_HIGH: "#D35400",      # Deep Orange
    PRIORITY_CRITICAL: "#C0392B"   # Crimson Red
}

STATUS_COLORS = {
    STATUS_SUBMITTED: "#6C757D",
    STATUS_UNDER_REVIEW: "#17A2B8",
    STATUS_VERIFIED: "#007BFF",
    STATUS_ASSIGNED: "#6610F2",
    STATUS_IN_PROGRESS: "#FD7E14",
    STATUS_RESOLVED: "#28A745",
    STATUS_REJECTED: "#DC3545"
}

# Duplicate Detection Configuration
DUPLICATE_SIMILARITY_THRESHOLD = 0.65  # 65% cosine similarity

# Priority Weight Factors
SEVERITY_WEIGHT = 0.35
SAFETY_RISK_WEIGHT = 0.35
PUBLIC_IMPACT_WEIGHT = 0.30
AGING_BONUS_PER_DAY = 2.0  # Points added per waiting day (max 20)

# Severity Options
SEVERITY_LEVELS = {
    "Minor": 1,
    "Moderate": 2,
    "Major": 3,
    "Severe": 4
}

# Safety Risk Options
SAFETY_RISK_LEVELS = {
    "No Immediate Hazard": 1,
    "Low Hazard (Caution Advised)": 2,
    "Moderate Hazard (Injury Risk)": 3,
    "Life-Threatening / Severe Hazard": 4
}

# Public Impact Options
PUBLIC_IMPACT_LEVELS = {
    "Isolated (Few Households)": 1,
    "Local Street / Neighborhood": 2,
    "Ward / Commercial Market Area": 3,
    "Arterial Road / Major Public Transit Corridor": 4
}

# Custom CSS for Navy Blue & White Smart City Theme
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* Smart City Header Banner */
    .portal-header {
        background: linear-gradient(135deg, #0A192F 0%, #1B365D 50%, #0D47A1 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(10, 25, 47, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .portal-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF;
    }

    .portal-subtitle {
        font-size: 13px;
        color: #90CAF9;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Professional Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        text-align: left;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(10, 25, 47, 0.08);
    }

    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #0A192F;
        margin: 8px 0 4px 0;
    }

    .metric-desc {
        font-size: 12px;
        color: #94A3B8;
    }

    /* Priority Badges */
    .badge-priority-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }

    .badge-priority-medium {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }

    .badge-priority-high {
        background-color: #FBE9E7;
        color: #D84315;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }

    .badge-priority-critical {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 12px;
        display: inline-block;
        animation: pulse 2s infinite;
    }

    /* Duplicate Warning Alert Box */
    .duplicate-alert {
        background-color: #FFF8E1;
        border-left: 5px solid #FFA000;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #5D4037;
    }

    .ai-pill {
        background: #E3F2FD;
        color: #0D47A1;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #BBDEFB;
    }

    /* Buttons styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }

    .stButton>button[kind="primary"] {
        background-color: #0D47A1;
        color: white;
        border: none;
    }

    .stButton>button[kind="primary"]:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 12px rgba(13, 71, 161, 0.25);
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    .sidebar-user-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 20px;
    }
</style>
"""
