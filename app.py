"""
CivicAI – Intelligent Civic Complaint Prioritization System
Main Streamlit Application
Roles: Citizen Portal & Municipal Admin / Staff Portal
"""

import os
import datetime
from pathlib import Path
import streamlit as st
import numpy as np

# Page Configuration - Must be the first Streamlit command
st.set_page_config(
    page_title="CivicAI – Smart City Complaint Management",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import local architectural modules
import config
from config import (
    APP_NAME,
    APP_TAGLINE,
    MUNICIPAL_NAME,
    CATEGORIES,
    CATEGORY_ICONS,
    CATEGORY_DESCRIPTIONS,
    COMPLAINT_STATUSES,
    PRIORITY_LEVELS,
    PRIORITY_COLORS,
    STATUS_COLORS,
    SEVERITY_LEVELS,
    SAFETY_RISK_LEVELS,
    PUBLIC_IMPACT_LEVELS,
    CUSTOM_CSS,
    UPLOADS_DIR
)
import database
from database import (
    init_db,
    insert_complaint,
    get_all_complaints,
    get_complaint_by_id,
    get_complaints_by_user,
    update_complaint_status,
    assign_complaint_to_staff,
    update_admin_notes,
    get_status_history,
    get_all_staff,
    add_staff,
    get_analytics_metrics,
    get_category_counts,
    get_priority_counts,
    get_monthly_trends
)
import auth
from auth import (
    init_session_state,
    set_current_user,
    logout_user,
    login_user,
    register_citizen,
    validate_registration_inputs
)
import ai_model
from ai_model import get_classifier_pipeline
import priority_engine
from priority_engine import calculate_priority_score
import duplicate_detector
from duplicate_detector import DuplicateDetector
import seed_data

# Inject custom theme CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session and Seed Data if needed
init_session_state()
init_db()

# Ensure model and demo data are initialized
@st.cache_resource
def bootstrap_system():
    try:
        seed_data.seed_database()
    except Exception as e:
        pass
    return get_classifier_pipeline()

pipeline = bootstrap_system()
detector = DuplicateDetector()

# --- Helper UI Components ---

def render_portal_header(role_label="Citizen Services"):
    """Renders the top branding banner."""
    current_time = datetime.datetime.now().strftime("%A, %d %B %Y")
    st.markdown(f"""
    <div class="portal-header">
        <div>
            <div class="portal-title">🏛️ {APP_NAME}</div>
            <div class="portal-subtitle">{MUNICIPAL_NAME} • {APP_TAGLINE}</div>
        </div>
        <div style="text-align: right;">
            <span class="ai-pill">System Status: Online</span>
            <div style="font-size: 12px; color: #BBDEFB; margin-top: 4px;">📅 {current_time}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_priority_badge(level: str):
    """HTML Badge for Priority Level."""
    lvl = level.lower()
    return f'<span class="badge-priority-{lvl}">{level.upper()}</span>'

def render_status_badge(status: str):
    """HTML Badge for Complaint Status."""
    color = STATUS_COLORS.get(status, "#6C757D")
    return f'<span style="background-color: {color}20; color: {color}; border: 1px solid {color}50; padding: 4px 10px; border-radius: 12px; font-weight: 700; font-size: 12px;">{status}</span>'

# --- Navigation & Sidebar ---

with st.sidebar:
    st.markdown(f"### 🏢 {APP_NAME} Portal")
    st.caption("Intelligent Civic Prioritization Engine")

    # User Profile / Auth State Widget
    if st.session_state.authenticated:
        user = st.session_state.user
        role_badge = "👑 Municipal Admin" if user["role"] == "admin" else "👤 Verified Citizen"
        st.markdown(f"""
        <div class="sidebar-user-card">
            <div style="font-size: 11px; color: #0D47A1; font-weight: 700;">LOGGED IN AS</div>
            <div style="font-size: 15px; font-weight: 700; color: #0A192F;">{user['full_name']}</div>
            <div style="font-size: 12px; color: #64748B;">{user['email']}</div>
            <div style="margin-top: 6px;">
                <span class="ai-pill">{role_badge}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Log Out", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        st.info("👋 Welcome! Please log in or register below to report issues or access municipal staff administration.")

    st.markdown("---")
    
    # Portal Navigation Selection
portal_mode = st.radio(
    "Select Portal View:",
    [
        "👤 Citizen Portal (Demo View)",
        "🏛️ Municipal Admin Portal"
    ],
    index=0
)
st.markdown("---")
st.markdown("#### ⚡ System Highlights")
st.markdown("""
- **TF-IDF + Logistic Regression** AI Category Classifier
- **Dynamic Multi-Factor** Priority Scoring (Severity, Risk, Impact, Aging)
- **TF-IDF Cosine Similarity** Duplicate Detection
- **Permanent SQLite** Audit Logging
""")

st.markdown("---")
st.caption("CivicAI v1.0 • B.Tech AI Capstone System")


# ==============================================================================
# CITIZEN PORTAL WORKFLOW
# ==============================================================================

def render_citizen_portal():
    render_portal_header(role_label="Citizen Services")

    # If citizen is not logged in, show Auth Screen (Login / Register)
    if not st.session_state.authenticated:
        tab_login, tab_register = st.tabs(["🔑 Citizen Login", "📝 New Citizen Registration"])

        with tab_login:
            st.markdown("#### Access Citizen Redressal Portal")
            st.markdown("Login with your registered username or email to submit and track civic complaints.")

            col_l1, col_l2 = st.columns([1, 1])
            with col_l1:
                login_id = st.text_input("Username or Email", placeholder="e.g. citizen or citizen@example.com")
                login_pwd = st.text_input("Password", type="password", placeholder="Enter your password")

                if st.button("Log In as Citizen", type="primary", use_container_width=True):
                    if not login_id or not login_pwd:
                        st.error("Please enter both username/email and password.")
                    else:
                        user_obj, err = login_user(login_id, login_pwd)
                        if err:
                            st.error(f"❌ {err}")
                        else:
                            set_current_user(user_obj)
                            st.success(f"Welcome back, {user_obj['full_name']}!")
                            st.rerun()

            with col_l2:
                st.markdown("""
                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 10px;">
                    <div style="font-weight: 700; color: #0A192F; margin-bottom: 8px;">🔑 Demo Citizen Credentials:</div>
                    <div style="font-size: 13px; color: #334155; line-height: 1.6;">
                        • <b>Username:</b> <code>citizen</code><br>
                        • <b>Password:</b> <code>Citizen@123</code><br>
                        • <b>Role:</b> Citizen Resident
                    </div>
                    <div style="margin-top: 15px; font-size: 12px; color: #64748B;">
                        Or click the <b>New Citizen Registration</b> tab to create your own account instantly.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with tab_register:
            st.markdown("#### Register as a Smart City Citizen")
            st.markdown("Create a permanent profile to submit complaints, upload photographs, and track resolutions.")

            with st.form("reg_form"):
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    reg_name = st.text_input("Full Name *", placeholder="e.g. Ramesh Kumar")
                    reg_username = st.text_input("Username *", placeholder="e.g. ramesh_k")
                    reg_phone = st.text_input("Mobile Phone (Optional)", placeholder="e.g. 9876543210")

                with col_r2:
                    reg_email = st.text_input("Email Address *", placeholder="e.g. ramesh@example.com")
                    reg_pwd = st.text_input("Password *", type="password", placeholder="Minimum 6 characters")
                    reg_pwd_confirm = st.text_input("Confirm Password *", type="password", placeholder="Repeat password")

                reg_submit = st.form_submit_button("Complete Registration", type="primary", use_container_width=True)

                if reg_submit:
                    valid, err_msg = validate_registration_inputs(
                        reg_username, reg_email, reg_pwd, reg_pwd_confirm, reg_name, reg_phone
                    )
                    if not valid:
                        st.error(f"❌ {err_msg}")
                    else:
                        uid, reg_err = register_citizen(reg_username, reg_email, reg_pwd, reg_name, reg_phone)
                        if reg_err:
                            st.error(f"❌ {reg_err}")
                        else:
                            st.success("🎉 Registration successful! You may now log in using your credentials.")

        return

    # Authenticated Citizen Tabs
    user = st.session_state.user
    citizen_tabs = st.tabs([
        "📝 Submit Complaint",
        "🔍 Track Complaint Status",
        "📂 My Complaint History"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: SUBMIT COMPLAINT
    # --------------------------------------------------------------------------
    with citizen_tabs[0]:
        st.markdown("### 📢 Report a Civic Problem")
        st.markdown("Provide details about the issue. Our AI engine will automatically classify the category, evaluate safety risks, and calculate a prioritized response score.")

        with st.container():
            col_in1, col_in2 = st.columns([1.6, 1.0])

            with col_in1:
                comp_title = st.text_input(
                    "Complaint Title *",
                    placeholder="e.g. Massive pothole on MG Road causing two-wheeler accidents",
                    help="Provide a concise summary of the problem."
                )

                comp_desc = st.text_area(
                    "Detailed Description *",
                    height=130,
                    placeholder="Describe the issue, exact landmark, how long it has persisted, and the danger it poses...",
                    help="Include specific details for municipal staff and AI categorization."
                )

                comp_location = st.text_input(
                    "Location / Landmark / Ward *",
                    placeholder="e.g. MG Road, Near Metro Pillar 142, Ward 12",
                    help="Mention street, cross road, nearest landmark or ward."
                )

                comp_photo = st.file_uploader(
                    "Upload Optional Photograph (JPEG/PNG)",
                    type=["jpg", "jpeg", "png"],
                    help="Visual proof helps municipal engineers inspect the site faster."
                )

            with col_in2:
                st.markdown("#### 🎯 Priority Factors")
                st.caption("Citizen self-assessment parameters used by the prioritization algorithm:")

                severity_choice = st.selectbox(
                    "Severity Level *",
                    list(SEVERITY_LEVELS.keys()),
                    index=1,
                    help="Minor: Cosmetic/Inconvenience | Severe: Total failure/Destruction"
                )

                risk_choice = st.selectbox(
                    "Public Safety Hazard *",
                    list(SAFETY_RISK_LEVELS.keys()),
                    index=1,
                    help="Hazard to pedestrians, commuters, children, or property."
                )

                impact_choice = st.selectbox(
                    "Public Impact Scope *",
                    list(PUBLIC_IMPACT_LEVELS.keys()),
                    index=1,
                    help="Number of people affected by this civic breakdown."
                )

        # AI Real-time Preview Section
        if comp_title.strip() and comp_desc.strip():
            st.markdown("---")
            st.markdown("#### 🤖 AI Analysis & Prioritization Preview")

            # Run AI Inference
            ai_res = pipeline.predict_one(comp_title, comp_desc)
            pred_cat = ai_res["category"]
            pred_conf = ai_res["confidence"]
            pred_conf_pct = ai_res["confidence_pct"]

            # Run Priority Calculation
            p_calc = calculate_priority_score(
                severity=severity_choice,
                safety_risk=risk_choice,
                public_impact=impact_choice
            )
            p_score = p_calc["priority_score"]
            p_level = p_calc["priority_level"]

            # Run Duplicate Check
            dup_res = detector.check_for_duplicates(
                new_title=comp_title,
                new_description=comp_desc,
                new_location=comp_location
            )

            # Display AI Cards
            col_ai1, col_ai2, col_ai3 = st.columns(3)

            with col_ai1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">AI PREDICTED CATEGORY</div>
                    <div class="metric-value" style="font-size: 22px;">{CATEGORY_ICONS.get(pred_cat, '📌')} {pred_cat}</div>
                    <div class="metric-desc">Confidence: <b>{pred_conf_pct}%</b> (Logistic Regression)</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ai2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PRIORITY SCORE</div>
                    <div class="metric-value">{p_score} <span style="font-size: 16px; color: #64748B;">/ 100</span></div>
                    <div class="metric-desc">Level: {render_priority_badge(p_level)}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_ai3:
                dup_text = "⚠️ Duplicate Detected" if dup_res["is_duplicate"] else "✅ Unique Complaint"
                dup_color = "#D9534F" if dup_res["is_duplicate"] else "#28A745"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">DUPLICATE CHECK</div>
                    <div class="metric-value" style="font-size: 20px; color: {dup_color};">{dup_text}</div>
                    <div class="metric-desc">Cosine Similarity: <b>{dup_res['highest_similarity_pct']}%</b></div>
                </div>
                """, unsafe_allow_html=True)

            # Duplicate Warning Banner if threshold exceeded
            if dup_res["is_duplicate"] and dup_res["matched_complaint"]:
                match = dup_res["matched_complaint"]
                st.markdown(f"""
                <div class="duplicate-alert">
                    <b>⚠️ Potential Duplicate Alert ({dup_res['highest_similarity_pct']}% Similarity):</b><br>
                    A very similar complaint <b>#{match['complaint_id']}</b> ("{match['title']}") was already filed for <i>{match['location']}</i>.<br>
                    Current Status: <b>{match['status']}</b>. Your submission will be recorded and linked to expedite municipal resolution.
                </div>
                """, unsafe_allow_html=True)

            # Allow Citizen to confirm or change Category
            cat_choice = st.selectbox(
                "Confirm or Select Final Complaint Category:",
                CATEGORIES,
                index=CATEGORIES.index(pred_cat) if pred_cat in CATEGORIES else 0,
                help="You may override the AI prediction if your complaint fits another civic division."
            )

            # Submit Complaint Button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Register Official Civic Complaint", type="primary", use_container_width=True):
                # Save photo if uploaded
                saved_photo_path = None
                if comp_photo is not None:
                    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_ext = Path(comp_photo.name).suffix or ".jpg"
                    photo_filename = f"photo_{timestamp}_{user['id']}{file_ext}"
                    dest = UPLOADS_DIR / photo_filename
                    with open(dest, "wb") as f:
                        f.write(comp_photo.getbuffer())
                    saved_photo_path = str(dest)

                complaint_payload = {
                    "user_id": user["id"],
                    "citizen_name": user["full_name"],
                    "citizen_email": user["email"],
                    "title": comp_title.strip(),
                    "description": comp_desc.strip(),
                    "location": comp_location.strip(),
                    "category_ai": pred_cat,
                    "category_confirmed": cat_choice,
                    "confidence_score": pred_conf,
                    "severity_level": severity_choice,
                    "safety_risk": risk_choice,
                    "public_impact": impact_choice,
                    "priority_score": p_score,
                    "priority_level": p_level,
                    "status": "Submitted",
                    "photo_path": saved_photo_path,
                    "is_duplicate": 1 if dup_res["is_duplicate"] else 0,
                    "duplicate_of_id": dup_res["matched_complaint"]["complaint_id"] if (dup_res["is_duplicate"] and dup_res["matched_complaint"]) else None,
                    "duplicate_score": dup_res["highest_similarity"]
                }

                new_cid = insert_complaint(complaint_payload)
                st.balloons()
                st.success(f"""
                ### ✅ Complaint Registered Successfully!
                **Your Unique Complaint Tracking ID is:** `{new_cid}`
                
                - **Category:** {cat_choice}
                - **Assigned Priority Score:** {p_score} / 100 ({p_level})
                - **Current Status:** Submitted (Awaiting Municipal Officer Verification)
                
                Please save this Complaint ID for tracking updates in the **Track Complaint Status** tab.
                """)
        else:
            st.info("💡 Enter both Title and Description above to trigger real-time AI classification, duplicate detection, and priority calculation.")

    # --------------------------------------------------------------------------
    # TAB 2: TRACK COMPLAINT STATUS
    # --------------------------------------------------------------------------
    with citizen_tabs[1]:
        st.markdown("### 🔍 Track Complaint Progress")
        st.markdown("Enter your unique Complaint ID (e.g. `CIVIC-2026-0001`) to view live inspection, assignment, and resolution status.")

        track_col1, track_col2 = st.columns([2, 1])
        with track_col1:
            search_cid = st.text_input("Enter Complaint ID", placeholder="CIVIC-2026-XXXX").strip()

        if search_cid:
            comp = get_complaint_by_id(search_cid)
            if not comp:
                st.error(f"❌ No complaint found with ID `{search_cid}`. Please check the ID and try again.")
            else:
                st.markdown("---")
                # Main Status Header Card
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 20px; font-weight: 800; color: #0A192F;">{comp['complaint_id']}</span>
                        <div>{render_status_badge(comp['status'])} {render_priority_badge(comp['priority_level'])}</div>
                    </div>
                    <div style="font-size: 18px; font-weight: 700; color: #1E293B; margin-bottom: 8px;">{comp['title']}</div>
                    <div style="font-size: 14px; color: #475569; margin-bottom: 14px; line-height: 1.5;">{comp['description']}</div>
                    <div style="display: flex; gap: 20px; font-size: 13px; color: #64748B; border-top: 1px solid #F1F5F9; padding-top: 12px;">
                        <div>📍 <b>Location:</b> {comp['location']}</div>
                        <div>🏷️ <b>Category:</b> {comp['category_confirmed']}</div>
                        <div>📅 <b>Filed On:</b> {comp['created_at']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Assigned Officer Details if assigned
                if comp.get("staff_name"):
                    st.markdown(f"""
                    <div style="background-color: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 16px; margin-top: 15px;">
                        <div style="font-weight: 700; color: #166534; font-size: 14px;">👷 Assigned Field Officer</div>
                        <div style="font-size: 13px; color: #14532D; margin-top: 4px;">
                            <b>{comp['staff_name']}</b> • {comp['staff_department']} (📞 {comp.get('staff_phone', 'Official Helpline')})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Photograph display if attached
                if comp.get("photo_path") and Path(comp["photo_path"]).exists():
                    st.markdown("#### 📷 Attached Photograph")
                    st.image(comp["photo_path"], width=400, caption=f"Photograph for {comp['complaint_id']}")

                # Status Progression Tracker
                st.markdown("#### 📊 Progress Flow")
                all_stages = ["Submitted", "Verified", "Assigned", "In Progress", "Resolved"]
                curr_status = comp["status"]

                if curr_status == "Rejected":
                    st.error("❌ This complaint was rejected by municipal review. See audit remarks below.")
                else:
                    curr_idx = all_stages.index(curr_status) if curr_status in all_stages else 0
                    stage_cols = st.columns(len(all_stages))
                    for i, stage in enumerate(all_stages):
                        with stage_cols[i]:
                            if i <= curr_idx:
                                st.success(f"✓ **{stage}**")
                            else:
                                st.caption(f"○ {stage}")

                # Audit Trail / Status History Table
                st.markdown("#### 🕒 Audit History & Official Notes")
                history = get_status_history(comp["complaint_id"])
                if history:
                    for h in history:
                        prev = f"from <i>{h['previous_status']}</i> " if h['previous_status'] else ""
                        st.markdown(f"""
                        <div style="padding: 10px 14px; border-left: 3px solid #0D47A1; background: #F8FAFC; margin-bottom: 8px; border-radius: 4px;">
                            <div style="font-size: 13px; font-weight: 700; color: #0A192F;">
                                Status updated {prev}to <b>{h['new_status']}</b> by {h['changed_by']}
                            </div>
                            <div style="font-size: 12px; color: #64748B;">{h['timestamp']} • Remarks: {h.get('remarks') or 'None'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No history events logged yet.")

    # --------------------------------------------------------------------------
    # TAB 3: MY COMPLAINT HISTORY
    # --------------------------------------------------------------------------
    with citizen_tabs[2]:
        st.markdown("### 📂 My Complaint History")
        my_complaints = get_complaints_by_user(user["id"])

        if not my_complaints:
            st.info("You haven't submitted any complaints yet. Use the **Submit Complaint** tab to file your first report.")
        else:
            st.markdown(f"Total Registered Complaints: **{len(my_complaints)}**")

            # Filter options
            cat_filter = st.selectbox("Filter by Category", ["All"] + CATEGORIES, key="my_cat_filter")
            filtered_my = [
                c for c in my_complaints
                if cat_filter == "All" or c["category_confirmed"] == cat_filter
            ]

            for item in filtered_my:
                with st.expander(f"📌 {item['complaint_id']} – {item['title']} ({item['status']})"):
                    col_m1, col_m2 = st.columns([2, 1])
                    with col_m1:
                        st.markdown(f"**Description:** {item['description']}")
                        st.markdown(f"**Location:** {item['location']}")
                        st.markdown(f"**Category:** {item['category_confirmed']} (AI: {item['category_ai']} with {round(item['confidence_score']*100, 1)}% confidence)")
                    with col_m2:
                        st.markdown(f"**Priority Score:** {item['priority_score']} / 100 ({render_priority_badge(item['priority_level'])})", unsafe_allow_html=True)
                        st.markdown(f"**Status:** {render_status_badge(item['status'])}", unsafe_allow_html=True)
                        st.markdown(f"**Filed Date:** {item['created_at']}")


# ==============================================================================
# MUNICIPAL ADMIN & STAFF PORTAL WORKFLOW
# ==============================================================================

def render_admin_portal():
    render_portal_header(role_label="Municipal Staff Portal")

    # If not logged in as admin, show Admin Authentication Form
    if not (st.session_state.authenticated and st.session_state.role == "admin"):
        st.markdown("### 🔐 Municipal Staff & Administrator Login")
        st.markdown("Restricted access for zonal engineers, municipal officers, and system administrators.")

        col_a1, col_a2 = st.columns([1, 1])
        with col_a1:
            with st.form("admin_login_form"):
                adm_user = st.text_input("Officer Username / Email", placeholder="admin")
                adm_pass = st.text_input("Password", type="password", placeholder="Enter admin password")
                adm_submit = st.form_submit_button("Access Admin Portal", type="primary", use_container_width=True)

                if adm_submit:
                    u, err = login_user(adm_user, adm_pass)
                    if err:
                        st.error(f"❌ {err}")
                    elif u["role"] != "admin":
                        st.error("❌ Access Denied: This account does not possess Municipal Administrator privileges.")
                    else:
                        set_current_user(u)
                        st.success(f"Welcome, Officer {u['full_name']}!")
                        st.rerun()

        with col_a2:
            st.markdown("""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 20px; border-radius: 10px;">
                <div style="font-weight: 700; color: #0A192F; margin-bottom: 8px;">🔑 Default Municipal Credentials:</div>
                <div style="font-size: 13px; color: #334155; line-height: 1.6;">
                    • <b>Username:</b> <code>admin</code><br>
                    • <b>Password:</b> <code>Admin@123</code><br>
                    • <b>Access Level:</b> Full Municipal Zonal Authority
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    # Authenticated Admin Workspace
    admin_user = st.session_state.user
    admin_tabs = st.tabs([
        "📊 Executive Analytics",
        "📋 Complaints Management",
        "⚠️ Potential Duplicates",
        "👷 Municipal Staff"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: EXECUTIVE ANALYTICS
    # --------------------------------------------------------------------------
    with admin_tabs[0]:
        st.markdown("### 📊 Smart City Complaint Intelligence Dashboard")
        metrics = get_analytics_metrics()

        # KPI Metric Cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

        with col_kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">TOTAL COMPLAINTS</div>
                <div class="metric-value">{metrics['total']}</div>
                <div class="metric-desc">Lifetime registrations</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">PENDING ACTION</div>
                <div class="metric-value" style="color: #D35400;">{metrics['pending']}</div>
                <div class="metric-desc">Awaiting resolution</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RESOLVED ISSUES</div>
                <div class="metric-value" style="color: #28A745;">{metrics['resolved']}</div>
                <div class="metric-desc">Successfully redressed</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CRITICAL PRIORITY</div>
                <div class="metric-value" style="color: #C0392B;">{metrics['critical']}</div>
                <div class="metric-desc">Urgent safety risks</div>
            </div>
            """, unsafe_allow_html=True)

        with col_kpi5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">DUPLICATE ALERTS</div>
                <div class="metric-value" style="color: #6A1B9A;">{metrics['duplicates']}</div>
                <div class="metric-desc">Cosine matched (≥65%)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Category and Priority Distribution Visualizations
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### 📂 Complaints by Civic Category")
            cat_counts = get_category_counts()
            if cat_counts:
                # Build native Streamlit / Plotly bar chart
                import plotly.graph_objects as go
                fig_cat = go.Figure(data=[
                    go.Bar(
                        x=list(cat_counts.keys()),
                        y=list(cat_counts.values()),
                        marker_color="#0D47A1",
                        text=list(cat_counts.values()),
                        textposition="auto"
                    )
                ])
                fig_cat.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    xaxis_title="Category",
                    yaxis_title="Volume",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("No complaint data recorded yet.")

        with chart_col2:
            st.markdown("#### 🎯 Complaints by Priority Level")
            pri_counts = get_priority_counts()
            if pri_counts:
                import plotly.graph_objects as go
                colors = [PRIORITY_COLORS.get(p, "#6C757D") for p in pri_counts.keys()]
                fig_pri = go.Figure(data=[
                    go.Pie(
                        labels=list(pri_counts.keys()),
                        values=list(pri_counts.values()),
                        hole=0.45,
                        marker=dict(colors=colors)
                    )
                ])
                fig_pri.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig_pri, use_container_width=True)
            else:
                st.info("No priority data recorded yet.")

        # Monthly Trends
        st.markdown("#### 📅 Monthly Inflow & Resolution Volume")
        monthly_data = get_monthly_trends()
        if monthly_data:
            import plotly.graph_objects as go
            months = [m["month"] for m in monthly_data]
            totals = [m["total"] for m in monthly_data]
            resolved = [m["resolved"] for m in monthly_data]

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=months, y=totals, name="Total Filed", line=dict(color="#0D47A1", width=3)))
            fig_trend.add_trace(go.Scatter(x=months, y=resolved, name="Resolved", line=dict(color="#28A745", width=3)))
            fig_trend.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=260,
                xaxis_title="Month",
                yaxis_title="Count",
                template="plotly_white"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.caption("Insufficient historical data for monthly trend rendering.")

    # --------------------------------------------------------------------------
    # TAB 2: COMPLAINTS MANAGEMENT
    # --------------------------------------------------------------------------
    with admin_tabs[1]:
        st.markdown("### 📋 Municipal Complaint Redressal & Action Center")

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            flt_status = st.selectbox("Status Filter", ["All"] + COMPLAINT_STATUSES)
        with f_col2:
            flt_cat = st.selectbox("Category Filter", ["All"] + CATEGORIES)
        with f_col3:
            flt_pri = st.selectbox("Priority Filter", ["All"] + PRIORITY_LEVELS)

        all_comp = get_all_complaints(
            status_filter=flt_status,
            category_filter=flt_cat,
            priority_filter=flt_pri
        )

        st.markdown(f"Showing **{len(all_comp)}** complaints matching criteria (Sorted by Priority Score):")

        if not all_comp:
            st.info("No complaints match the selected filter criteria.")
        else:
            # Action Panel for Selected Complaint
            comp_options = {
                f"{c['complaint_id']} – {c['title']} ({c['status']}) [Score: {c['priority_score']}]": c["complaint_id"]
                for c in all_comp
            }
            selected_comp_key = st.selectbox("Select Complaint to Inspect & Action:", list(comp_options.keys()))
            sel_cid = comp_options[selected_comp_key]
            comp_detail = get_complaint_by_id(sel_cid)

            if comp_detail:
                st.markdown("---")
                det_col1, det_col2 = st.columns([1.5, 1.0])

                with det_col1:
                    st.markdown(f"#### 📌 Details: `{comp_detail['complaint_id']}`")
                    st.markdown(f"**Title:** {comp_detail['title']}")
                    st.markdown(f"**Description:** {comp_detail['description']}")
                    st.markdown(f"**Location:** {comp_detail['location']}")
                    st.markdown(f"**Citizen Contact:** {comp_detail['citizen_name']} ({comp_detail['citizen_email']})")

                    st.markdown("---")
                    st.markdown("##### 🤖 AI Model Telemetry")
                    st.markdown(f"""
                    - **AI Category Prediction:** `{comp_detail['category_ai']}`
                    - **Confidence Level:** `{round(comp_detail['confidence_score']*100, 1)}%`
                    - **Confirmed Category:** `{comp_detail['category_confirmed']}`
                    - **Calculated Priority:** `{comp_detail['priority_score']} / 100` ({comp_detail['priority_level']})
                    """)

                    if comp_detail.get("photo_path") and Path(comp_detail["photo_path"]).exists():
                        st.image(comp_detail["photo_path"], width=350, caption="Citizen Uploaded Photo")

                with det_col2:
                    st.markdown("#### ⚡ Municipal Action Panel")

                    # Current Status & Officer Badge
                    st.markdown(f"**Current Status:** {render_status_badge(comp_detail['status'])}", unsafe_allow_html=True)
                    if comp_detail.get("staff_name"):
                        st.markdown(f"**Assigned Officer:** 👷 {comp_detail['staff_name']} ({comp_detail['staff_department']})")

                    # Action 1: Assign to Municipal Staff
                    st.markdown("##### 👷 Assign to Field Officer")
                    staff_list = get_all_staff(active_only=True)
                    staff_opts = {f"{s['name']} – {s['department']}": s["id"] for s in staff_list}

                    selected_staff = st.selectbox("Select Officer:", list(staff_opts.keys()))
                    assign_notes = st.text_input("Special Field Instructions:", placeholder="e.g. Inspect site by 2 PM")

                    if st.button("Assign Officer & Mark 'Assigned'", type="primary", use_container_width=True):
                        s_id = staff_opts[selected_staff]
                        ok, err = assign_complaint_to_staff(
                            complaint_id=comp_detail["complaint_id"],
                            staff_id=s_id,
                            assigned_by=f"Admin: {admin_user['full_name']}",
                            instructions=assign_notes
                        )
                        if ok:
                            st.success(f"Complaint assigned to {selected_staff}!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                    # Action 2: Update Complaint Status
                    st.markdown("##### 🔄 Change Resolution Status")
                    new_st = st.selectbox(
                        "Update Status to:",
                        COMPLAINT_STATUSES,
                        index=COMPLAINT_STATUSES.index(comp_detail["status"]) if comp_detail["status"] in COMPLAINT_STATUSES else 0
                    )
                    status_remark = st.text_input("Status Change Remarks:", placeholder="e.g. Repair crew dispatched with asphalt")

                    if st.button("Apply Status Change", use_container_width=True):
                        ok, err = update_complaint_status(
                            complaint_id=comp_detail["complaint_id"],
                            new_status=new_st,
                            changed_by=f"Admin: {admin_user['full_name']}",
                            remarks=status_remark
                        )
                        if ok:
                            st.success(f"Status updated to '{new_st}'!")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")

                    # Action 3: Admin Internal Notes
                    st.markdown("##### 📝 Internal Municipal Notes")
                    curr_notes = comp_detail.get("admin_notes") or ""
                    new_notes = st.text_area("Administrative Notes:", value=curr_notes, height=80)
                    if st.button("Save Admin Notes", use_container_width=True):
                        update_admin_notes(comp_detail["complaint_id"], new_notes)
                        st.success("Notes saved permanently.")
                        st.rerun()

    # --------------------------------------------------------------------------
    # TAB 3: POTENTIAL DUPLICATES
    # --------------------------------------------------------------------------
    with admin_tabs[2]:
        st.markdown("### ⚠️ Potential Duplicate Complaints Inspector")
        st.markdown("CivicAI automatically detects matching complaints filed across the same ward or description using **TF-IDF Vectorization** and **Cosine Similarity**.")

        dups = [c for c in get_all_complaints() if c["is_duplicate"] == 1]
        if not dups:
            st.success("✅ No duplicate complaints currently flagged in the system.")
        else:
            st.markdown(f"Found **{len(dups)}** complaints flagged with high similarity ($\ge 65\%$):")
            for d in dups:
                orig_id = d.get("duplicate_of_id")
                orig = get_complaint_by_id(orig_id) if orig_id else None

                with st.expander(f"⚠️ {d['complaint_id']} (Sim: {round(d['duplicate_score']*100, 1)}% with #{orig_id or 'Unknown'})"):
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.markdown(f"#### Duplicate Candidate: `{d['complaint_id']}`")
                        st.markdown(f"**Title:** {d['title']}")
                        st.markdown(f"**Description:** {d['description']}")
                        st.markdown(f"**Location:** {d['location']}")
                        st.markdown(f"**Status:** {d['status']}")
                        st.markdown(f"**Filed On:** {d['created_at']}")

                    with col_d2:
                        st.markdown(f"#### Original Complaint: `{orig_id}`")
                        if orig:
                            st.markdown(f"**Title:** {orig['title']}")
                            st.markdown(f"**Description:** {orig['description']}")
                            st.markdown(f"**Location:** {orig['location']}")
                            st.markdown(f"**Status:** {orig['status']}")
                            st.markdown(f"**Assigned Officer:** {orig.get('staff_name', 'None')}")
                        else:
                            st.caption("Original complaint record not found.")

                    st.markdown("---")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button(f"Mark #{d['complaint_id']} as Duplicate & Close", key=f"close_{d['complaint_id']}"):
                            update_complaint_status(
                                d["complaint_id"],
                                "Resolved",
                                f"Admin: {admin_user['full_name']}",
                                f"Merged and closed as duplicate of #{orig_id}"
                            )
                            st.success(f"Complaint #{d['complaint_id']} closed as duplicate.")
                            st.rerun()

                    with col_b2:
                        if st.button(f"Disregard Warning (Keep #{d['complaint_id']} Unique)", key=f"uniq_{d['complaint_id']}"):
                            # Update complaint flag
                            conn = database.get_connection()
                            conn.execute("UPDATE complaints SET is_duplicate = 0 WHERE complaint_id = ?", (d["complaint_id"],))
                            conn.commit()
                            conn.close()
                            st.success(f"Complaint #{d['complaint_id']} marked as unique.")
                            st.rerun()

    # --------------------------------------------------------------------------
    # TAB 4: MUNICIPAL STAFF
    # --------------------------------------------------------------------------
    with admin_tabs[3]:
        st.markdown("### 👷 Municipal Staff & Field Engineers Directory")

        col_st1, col_st2 = st.columns([1.6, 1.0])
        with col_st1:
            st.markdown("#### Registered Municipal Officers")
            all_staff_members = get_all_staff(active_only=False)
            if all_staff_members:
                for s in all_staff_members:
                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{s['name']}</b>
                            <span class="ai-pill">{s['department']}</span>
                        </div>
                        <div style="font-size: 12px; color: #64748B; margin-top: 4px;">
                            ✉️ {s['email']} • 📞 {s.get('phone', 'N/A')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No staff members registered.")

        with col_st2:
            st.markdown("#### ➕ Add New Municipal Officer")
            with st.form("add_staff_form"):
                ns_name = st.text_input("Officer Full Name *", placeholder="e.g. Suresh Gowda")
                ns_dept = st.selectbox("Department *", [
                    "Roads & Highways Department",
                    "Sanitation & Solid Waste Management",
                    "Water Supply & Potable Distribution",
                    "Stormwater & Drainage Division",
                    "Street Lighting & Electrical Division"
                ])
                ns_email = st.text_input("Government Email *", placeholder="e.g. suresh.g@smartcity.gov.in")
                ns_phone = st.text_input("Direct Phone", placeholder="e.g. 9845012349")
                ns_submit = st.form_submit_button("Register Field Officer", type="primary", use_container_width=True)

                if ns_submit:
                    if not ns_name.strip() or not ns_email.strip():
                        st.error("Name and email are required.")
                    else:
                        sid, serr = add_staff(ns_name, ns_dept, ns_email, ns_phone)
                        if serr:
                            st.error(f"Error: {serr}")
                        else:
                            st.success(f"Officer {ns_name} registered successfully!")
                            st.rerun()


# ==============================================================================
# MAIN ROUTING CONTROLLER
# ==============================================================================

if "Municipal Admin" in portal_mode:
    render_admin_portal()
else:
    render_citizen_portal()
