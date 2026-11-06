"""
CivicAI – SQLite Database Layer
Handles permanent storage, relational schema, CRUD operations, and initial seeding.
"""

import sqlite3
import datetime
from pathlib import Path
from config import DB_PATH, UPLOADS_DIR

def get_connection():
    """Create and return thread-safe SQLite connection with Row factory."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        phone TEXT,
        role TEXT NOT NULL CHECK(role IN ('citizen', 'admin')),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. Staff Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS staff (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        phone TEXT,
        active INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 3. Complaints Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT UNIQUE NOT NULL,
        user_id INTEGER NOT NULL,
        citizen_name TEXT NOT NULL,
        citizen_email TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        location TEXT NOT NULL,
        category_ai TEXT NOT NULL,
        category_confirmed TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        severity_level TEXT NOT NULL,
        safety_risk TEXT NOT NULL,
        public_impact TEXT NOT NULL,
        priority_score INTEGER NOT NULL,
        priority_level TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'Submitted',
        photo_path TEXT,
        is_duplicate INTEGER DEFAULT 0,
        duplicate_of_id TEXT,
        duplicate_score REAL DEFAULT 0.0,
        assigned_staff_id INTEGER,
        admin_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id),
        FOREIGN KEY (assigned_staff_id) REFERENCES staff (id)
    )
    """)

    # 4. Assignments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS assignments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT NOT NULL,
        staff_id INTEGER NOT NULL,
        assigned_by TEXT NOT NULL,
        instructions TEXT,
        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (complaint_id) REFERENCES complaints (complaint_id),
        FOREIGN KEY (staff_id) REFERENCES staff (id)
    )
    """)

    # 5. Status History Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS status_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        complaint_id TEXT NOT NULL,
        previous_status TEXT,
        new_status TEXT NOT NULL,
        changed_by TEXT NOT NULL,
        remarks TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (complaint_id) REFERENCES complaints (complaint_id)
    )
    """)

    conn.commit()
    conn.close()

def generate_next_complaint_id():
    """Generate human-readable serial ID like CIVIC-2026-0001."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM complaints")
    count = cursor.fetchone()[0]
    conn.close()
    year = datetime.datetime.now().year
    return f"CIVIC-{year}-{count + 1:04d}"

# --- User Operations ---

def create_user(username, email, password_hash, full_name, phone=None, role="citizen"):
    """Register a new user (citizen or admin)."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO users (username, email, password_hash, full_name, phone, role)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (username.strip().lower(), email.strip().lower(), password_hash, full_name.strip(), phone, role))
        conn.commit()
        user_id = cursor.lastrowid
        return user_id, None
    except sqlite3.IntegrityError as e:
        if "users.username" in str(e):
            return None, "Username is already registered. Please choose another."
        elif "users.email" in str(e):
            return None, "Email is already registered. Please login or use another email."
        return None, f"Database constraint error: {e}"
    finally:
        conn.close()

def get_user_by_username(username):
    """Retrieve user record by username."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_email(email):
    """Retrieve user record by email."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_user_by_id(user_id):
    """Retrieve user record by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# --- Staff Operations ---

def get_all_staff(active_only=True):
    """Retrieve list of municipal staff members."""
    conn = get_connection()
    cursor = conn.cursor()
    if active_only:
        cursor.execute("SELECT * FROM staff WHERE active = 1 ORDER BY department, name")
    else:
        cursor.execute("SELECT * FROM staff ORDER BY department, name")
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_staff_by_id(staff_id):
    """Retrieve single staff member by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM staff WHERE id = ?", (staff_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def add_staff(name, department, email, phone=None):
    """Add a new municipal staff member."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO staff (name, department, email, phone)
        VALUES (?, ?, ?, ?)
        """, (name, department, email, phone))
        conn.commit()
        return cursor.lastrowid, None
    except sqlite3.IntegrityError as e:
        return None, str(e)
    finally:
        conn.close()

# --- Complaint Operations ---

def insert_complaint(complaint_data):
    """Insert a new civic complaint and record initial status history."""
    conn = get_connection()
    cursor = conn.cursor()

    complaint_id = generate_next_complaint_id()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    INSERT INTO complaints (
        complaint_id, user_id, citizen_name, citizen_email,
        title, description, location, category_ai, category_confirmed,
        confidence_score, severity_level, safety_risk, public_impact,
        priority_score, priority_level, status, photo_path,
        is_duplicate, duplicate_of_id, duplicate_score, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        complaint_data["user_id"],
        complaint_data["citizen_name"],
        complaint_data["citizen_email"],
        complaint_data["title"],
        complaint_data["description"],
        complaint_data["location"],
        complaint_data["category_ai"],
        complaint_data["category_confirmed"],
        complaint_data["confidence_score"],
        complaint_data["severity_level"],
        complaint_data["safety_risk"],
        complaint_data["public_impact"],
        complaint_data["priority_score"],
        complaint_data["priority_level"],
        complaint_data.get("status", "Submitted"),
        complaint_data.get("photo_path"),
        complaint_data.get("is_duplicate", 0),
        complaint_data.get("duplicate_of_id"),
        complaint_data.get("duplicate_score", 0.0),
        now_str,
        now_str
    ))

    # Record Initial Status in History
    cursor.execute("""
    INSERT INTO status_history (complaint_id, previous_status, new_status, changed_by, remarks, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        None,
        complaint_data.get("status", "Submitted"),
        f"Citizen ({complaint_data['citizen_name']})",
        "Complaint registered through CivicAI citizen portal.",
        now_str
    ))

    conn.commit()
    conn.close()
    return complaint_id

def get_complaint_by_id(complaint_id):
    """Retrieve full details of a specific complaint by complaint_id."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT c.*, s.name as staff_name, s.department as staff_department, s.phone as staff_phone
    FROM complaints c
    LEFT JOIN staff s ON c.assigned_staff_id = s.id
    WHERE c.complaint_id = ?
    """, (complaint_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def get_complaints_by_user(user_id):
    """Retrieve all complaints submitted by a particular user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM complaints
    WHERE user_id = ?
    ORDER BY created_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_complaints(status_filter=None, category_filter=None, priority_filter=None):
    """Retrieve all complaints with optional filtering for Admin Portal."""
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT c.*, s.name as staff_name, s.department as staff_department
    FROM complaints c
    LEFT JOIN staff s ON c.assigned_staff_id = s.id
    WHERE 1=1
    """
    params = []

    if status_filter and status_filter != "All":
        query += " AND c.status = ?"
        params.append(status_filter)

    if category_filter and category_filter != "All":
        query += " AND c.category_confirmed = ?"
        params.append(category_filter)

    if priority_filter and priority_filter != "All":
        query += " AND c.priority_level = ?"
        params.append(priority_filter)

    query += " ORDER BY c.priority_score DESC, c.created_at DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_complaint_status(complaint_id, new_status, changed_by, remarks=""):
    """Update status of a complaint and append record to status history."""
    conn = get_connection()
    cursor = conn.cursor()

    # Get current status
    cursor.execute("SELECT status FROM complaints WHERE complaint_id = ?", (complaint_id,))
    current = cursor.fetchone()
    if not current:
        conn.close()
        return False, "Complaint not found."

    prev_status = current["status"]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
    UPDATE complaints
    SET status = ?, updated_at = ?
    WHERE complaint_id = ?
    """, (new_status, now_str, complaint_id))

    cursor.execute("""
    INSERT INTO status_history (complaint_id, previous_status, new_status, changed_by, remarks, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (complaint_id, prev_status, new_status, changed_by, remarks, now_str))

    conn.commit()
    conn.close()
    return True, None

def assign_complaint_to_staff(complaint_id, staff_id, assigned_by, instructions=""):
    """Assign complaint to a municipal staff officer and update status to 'Assigned'."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT status FROM complaints WHERE complaint_id = ?", (complaint_id,))
    current = cursor.fetchone()
    if not current:
        conn.close()
        return False, "Complaint not found."

    prev_status = current["status"]
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Update complaint
    cursor.execute("""
    UPDATE complaints
    SET assigned_staff_id = ?, status = 'Assigned', updated_at = ?
    WHERE complaint_id = ?
    """, (staff_id, now_str, complaint_id))

    # Log in assignments table
    cursor.execute("""
    INSERT INTO assignments (complaint_id, staff_id, assigned_by, instructions, assigned_at)
    VALUES (?, ?, ?, ?, ?)
    """, (complaint_id, staff_id, assigned_by, instructions, now_str))

    # Fetch staff name
    cursor.execute("SELECT name, department FROM staff WHERE id = ?", (staff_id,))
    staff = cursor.fetchone()
    staff_info = f"{staff['name']} ({staff['department']})" if staff else f"Staff ID #{staff_id}"

    # Log in status history
    remarks = f"Assigned to {staff_info}. {instructions}".strip()
    cursor.execute("""
    INSERT INTO status_history (complaint_id, previous_status, new_status, changed_by, remarks, timestamp)
    VALUES (?, ?, 'Assigned', ?, ?, ?)
    """, (complaint_id, prev_status, assigned_by, remarks, now_str))

    conn.commit()
    conn.close()
    return True, None

def update_admin_notes(complaint_id, notes):
    """Save administrative notes on a complaint."""
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    UPDATE complaints
    SET admin_notes = ?, updated_at = ?
    WHERE complaint_id = ?
    """, (notes, now_str, complaint_id))
    conn.commit()
    conn.close()

def get_status_history(complaint_id):
    """Retrieve complete audit trail and status timeline for a complaint."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT * FROM status_history
    WHERE complaint_id = ?
    ORDER BY timestamp ASC
    """, (complaint_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Analytics Queries ---

def get_analytics_metrics():
    """Compute KPI figures for dashboard cards."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM complaints")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status IN ('Submitted', 'Under Review', 'Verified', 'Assigned', 'In Progress')")
    pending = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE status = 'Resolved'")
    resolved = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE priority_level = 'Critical'")
    critical = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM complaints WHERE is_duplicate = 1")
    duplicates = cursor.fetchone()[0]

    conn.close()
    return {
        "total": total,
        "pending": pending,
        "resolved": resolved,
        "critical": critical,
        "duplicates": duplicates
    }

def get_category_counts():
    """Retrieve complaint count aggregated by confirmed category."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT category_confirmed, COUNT(*) as count
    FROM complaints
    GROUP BY category_confirmed
    ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return {r["category_confirmed"]: r["count"] for r in rows}

def get_priority_counts():
    """Retrieve complaint count aggregated by priority level."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT priority_level, COUNT(*) as count
    FROM complaints
    GROUP BY priority_level
    ORDER BY count DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return {r["priority_level"]: r["count"] for r in rows}

def get_monthly_trends():
    """Retrieve monthly complaint volume trends."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT strftime('%Y-%m', created_at) as month,
           COUNT(*) as total,
           SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved
    FROM complaints
    GROUP BY month
    ORDER BY month ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
