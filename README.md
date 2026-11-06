# CivicAI – Intelligent Civic Complaint Prioritization System

**CivicAI** is an AI-powered municipal complaint prioritization and redressal platform built for modern Smart Cities. It combines **Natural Language Processing (TF-IDF vectorization)** and **Multinomial Logistic Regression** with a **dynamic rule-based priority engine** and **cosine similarity duplicate detection** to automate the intake, categorization, and routing of civic complaints.

Developed as a **B.Tech AI Mini-Capstone Project**.

---

## 🏛️ Key Features

### 👤 Citizen Portal
- **Secure Registration & Login**: Multi-user authentication with PBKDF2-HMAC-SHA256 salted password hashing.
- **Smart Complaint Submission**:
  - Enter complaint Title, Detailed Description, and Location / Landmark.
  - Optional photograph upload (JPG/PNG).
  - Self-assessment inputs for **Severity**, **Public Safety Risk**, and **Public Impact**.
- **Real-time AI Prediction**:
  - Instant classification into one of 5 civic categories with a confidence score.
  - Displays full probability breakdown across all categories.
  - Allows citizen to verify or override the category.
- **Duplicate Complaint Warning**:
  - Real-time TF-IDF cosine similarity comparison against existing complaints in the database.
  - Alerts citizens if a similar issue was recently filed (≥65% similarity threshold).
- **Complaint Tracking**:
  - Receive a unique, human-readable serial ID (e.g., `CIVIC-2026-0001`).
  - Search by Complaint ID to view live progress stages (`Submitted` → `Verified` → `Assigned` → `In Progress` → `Resolved`).
  - View assigned municipal field officer details and complete chronological audit history.
- **My Complaint History**: View and filter past submissions.

---

### 👑 Municipal Staff / Admin Portal
- **Executive Smart City Analytics Dashboard**:
  - High-impact KPI cards: Total Complaints, Pending Redressals, Resolved Issues, Critical Priority cases, and Duplicate Alerts.
  - Interactive distribution charts: Complaints by Category, Complaints by Priority, and Monthly Inflow & Resolution Trends.
- **Complaints Management & Action Center**:
  - Searchable, filterable complaints table (by Status, Category, Priority).
  - Inspect full complaint details, citizen contact info, uploaded photo, and AI telemetry.
  - **Verify or Reject** complaints with justification remarks.
  - **Assign Complaints** to specific municipal field officers across specialized departments with custom instructions.
  - **Update Status** (`Under Review`, `Verified`, `Assigned`, `In Progress`, `Resolved`, `Rejected`).
  - Record internal administrative notes.
- **Duplicate Complaint Inspector**:
  - Side-by-side inspection of duplicate candidates with similarity scores.
  - One-click action to merge and close duplicates or mark as unique.
- **Field Staff Directory**:
  - Department-wise directory of registered engineers and zonal officers.
  - Form to onboard new municipal field staff.

---

## 🤖 AI & Mathematical Architecture

### 1. Complaint Categorization (5 Civic Categories)
1. **Road Damage** (Potholes, broken asphalt, craters, cave-ins)
2. **Garbage** (Overflowing dumpsters, illegal dumps, litter, uncollected waste)
3. **Water Leakage** (Pipeline bursts, continuous freshwater leaks, leaking valves)
4. **Blocked Drainage** (Clogged storm drains, sewage overflows, backed up culverts)
5. **Broken Streetlights** (Non-functional lamps, flickering poles, exposed live wiring)

### 2. TF-IDF Text Vectorization
Text from Title and Description is preprocessed (lowercased, non-alphanumeric stripped, stop words removed), unigrams and bigrams are extracted, and sublinear TF scaling is applied:
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d))$$
$$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
Vectors are normalized using the L2 norm: $\|\mathbf{x}\|_2 = 1$.

### 3. Logistic Regression Classifier
Multinomial Logistic Regression (Softmax Regression) computes calibrated posterior probabilities:
$$P(Y = k \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^T \mathbf{x} + b_k)}{\sum_{j=1}^K \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$
Trained with Mini-batch Gradient Descent with Nesterov Momentum and L2 regularization.

### 4. Duplicate Complaint Detection (Cosine Similarity)
Calculates cosine similarity between the incoming complaint vector $\mathbf{u}$ and existing complaint vectors $\mathbf{v}$:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Complaints with $\ge 65\%$ similarity are flagged as potential duplicates.

### 5. Multi-Factor Priority Scoring Engine
Computes an objective, transparent priority score (0 to 100):
$$\text{Base Score} = \left(0.35 \times \text{Severity} + 0.35 \times \text{Safety Risk} + 0.30 \times \text{Public Impact}\right) \times 25$$
$$\text{Aging Factor} = \min\left(20.0, \, \text{Days Pending} \times 2.0\right)$$
$$\text{Total Priority Score} = \min\left(100, \, \text{round}(\text{Base Score} + \text{Aging Factor})\right)$$

**Priority Tiers:**
- **0 – 39**: Low (Green)
- **40 – 64**: Medium (Amber)
- **65 – 84**: High (Orange)
- **85 – 100**: Critical (Crimson Red)

---

## 🗄️ Relational Database Schema (SQLite)

- **`users`**: User accounts (`id`, `username`, `email`, `password_hash`, `full_name`, `phone`, `role`, `created_at`).
- **`staff`**: Field officers (`id`, `name`, `department`, `email`, `phone`, `active`, `created_at`).
- **`complaints`**: Core records (`id`, `complaint_id`, `user_id`, `citizen_name`, `citizen_email`, `title`, `description`, `location`, `category_ai`, `category_confirmed`, `confidence_score`, `severity_level`, `safety_risk`, `public_impact`, `priority_score`, `priority_level`, `status`, `photo_path`, `is_duplicate`, `duplicate_of_id`, `duplicate_score`, `assigned_staff_id`, `admin_notes`, `created_at`, `updated_at`).
- **`assignments`**: Field officer dispatches (`id`, `complaint_id`, `staff_id`, `assigned_by`, `instructions`, `assigned_at`).
- **`status_history`**: Immutable audit trail (`id`, `complaint_id`, `previous_status`, `new_status`, `changed_by`, `remarks`, `timestamp`).

---

## 🔑 Pre-Configured Login Credentials

| Role | Username | Password | Email | Access |
|---|---|---|---|---|
| **Municipal Administrator** | `admin` | `Admin@123` | `admin@smartcity.gov.in` | Executive Dashboard, Complaints Action, Staff Assignment, Duplicates |
| **Demo Citizen Resident** | `citizen` | `Citizen@123` | `citizen@example.com` | Report Issues, AI Preview, Track Status, Complaint History |

*New citizens can also register their own account directly through the registration form.*

---

## 🚀 How to Run the Application

### 1. Prerequisites
Ensure Python 3.10+ (or Python 3.14) is installed.

### 2. Navigate to Project Directory
```powershell
cd C:\Users\garag\.gemini\antigravity\scratch\civic_ai
```

### 3. Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Initialize Database & Train AI Model
```powershell
python seed_data.py
```
*This generates the 350-sample synthetic dataset, trains the Logistic Regression classifier, seeds default users, staff, and demo complaints.*

### 6. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## 📁 Project Directory Structure

```
civic_ai/
├── app.py                      # Streamlit Multi-Role Web Application
├── config.py                   # Configuration constants, UI theme, priority weights
├── database.py                 # SQLite schema, queries, CRUD functions
├── auth.py                     # Authentication, salted PBKDF2 password hashing
├── ai_model.py                 # TF-IDF Vectorizer + Logistic Regression Classifier
├── duplicate_detector.py       # TF-IDF Cosine Similarity duplicate engine
├── priority_engine.py          # Rule-based priority scoring calculation
├── train_model.py              # 350-sample dataset generator & training script
├── seed_data.py                # Database initialization and seeder
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and usage guide
├── data/
│   ├── civic_complaints.db     # SQLite database
│   └── synthetic_complaints_350.csv  # 350-sample synthetic dataset
├── models/
│   ├── civic_classifier.joblib # Serialized Logistic Regression model
│   └── vectorizer.joblib       # Serialized TF-IDF vectorizer
└── uploads/                    # User photograph storage
```

---

## ⚠️ Academic Disclaimer
> **NOTE:** The 350-sample training dataset is synthetically generated for academic demonstration in a capstone environment. Validation accuracy on synthetic data reflects internal benchmark consistency and does not represent real-world deployment performance on unconstrained natural language complaints.
