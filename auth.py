"""
CivicAI – Authentication & Security Module
Provides salted cryptographic password hashing, input validation, and role-based session controls.
"""

import hashlib
import os
import re
import streamlit as st
from database import create_user, get_user_by_username, get_user_by_email

# Number of iterations for PBKDF2 hashing
HASH_ITERATIONS = 100_000

def hash_password(password: str) -> str:
    """
    Hash a plaintext password using PBKDF2-HMAC-SHA256 with a unique random 16-byte salt.
    Format stored: salt$hash (hex encoded).
    """
    salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        HASH_ITERATIONS
    )
    return f"{salt}${dk.hex()}"

def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    """
    Verify a provided password against the stored salt$hash string.
    """
    if not stored_password_hash or "$" not in stored_password_hash:
        return False
    try:
        salt, expected_hash = stored_password_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            provided_password.encode("utf-8"),
            bytes.fromhex(salt),
            HASH_ITERATIONS
        )
        return dk.hex() == expected_hash
    except Exception:
        return False

def validate_registration_inputs(username, email, password, confirm_password, full_name, phone=None):
    """
    Comprehensive input validation for new user registrations.
    Returns (is_valid: bool, error_message: str | None).
    """
    # Full Name Check
    if not full_name or len(full_name.strip()) < 2:
        return False, "Please enter a valid full name (at least 2 characters)."

    # Username Check
    username_clean = username.strip().lower()
    if not username_clean or len(username_clean) < 3 or len(username_clean) > 30:
        return False, "Username must be between 3 and 30 characters."
    if not re.match(r"^[a-zA-Z0-9_.-]+$", username_clean):
        return False, "Username may only contain letters, numbers, underscores, dots, or hyphens."

    # Email Check
    email_clean = email.strip().lower()
    email_regex = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    if not re.match(email_regex, email_clean):
        return False, "Please enter a valid email address (e.g., citizen@example.com)."

    # Phone Check (Optional)
    if phone and phone.strip():
        phone_clean = re.sub(r"[\s\-\(\)\+]", "", phone.strip())
        if not phone_clean.isdigit() or len(phone_clean) < 7 or len(phone_clean) > 15:
            return False, "Please enter a valid phone number (7 to 15 digits)."

    # Password Length & Match
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    if password != confirm_password:
        return False, "Passwords do not match. Please re-enter identical passwords."

    return True, None

def login_user(username_or_email, password):
    """
    Authenticate user by username or email.
    Returns (user_dict: dict | None, error_message: str | None).
    """
    query_str = username_or_email.strip().lower()
    user = get_user_by_username(query_str)
    if not user:
        user = get_user_by_email(query_str)

    if not user:
        return None, "Invalid username or email address."

    if not verify_password(user["password_hash"], password):
        return None, "Invalid password. Please check your credentials and try again."

    return user, None

def register_citizen(username, email, password, full_name, phone=None):
    """
    Register a new citizen in the system.
    """
    pwd_hash = hash_password(password)
    user_id, err = create_user(
        username=username,
        email=email,
        password_hash=pwd_hash,
        full_name=full_name,
        phone=phone,
        role="citizen"
    )
    return user_id, err

def register_admin(username, email, password, full_name, phone=None):
    """
    Register a new administrator/municipal officer.
    """
    pwd_hash = hash_password(password)
    user_id, err = create_user(
        username=username,
        email=email,
        password_hash=pwd_hash,
        full_name=full_name,
        phone=phone,
        role="admin"
    )
    return user_id, err

# --- Session Management in Streamlit ---

def init_session_state():
    """Ensure session state variables for authentication exist."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user" not in st.session_state:
        st.session_state.user = None
    if "role" not in st.session_state:
        st.session_state.role = None

def set_current_user(user_dict):
    """Store logged-in user in Streamlit session state."""
    st.session_state.authenticated = True
    st.session_state.user = user_dict
    st.session_state.role = user_dict["role"]

def logout_user():
    """Clear session state and log out."""
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.role = None
