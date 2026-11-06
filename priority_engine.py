"""
CivicAI – Dynamic Priority Scoring Engine
Calculates multi-factor priority scores based on severity, safety risk, public impact, and aging waiting time.
"""

import datetime
from config import (
    SEVERITY_WEIGHT,
    SAFETY_RISK_WEIGHT,
    PUBLIC_IMPACT_WEIGHT,
    AGING_BONUS_PER_DAY,
    SEVERITY_LEVELS,
    SAFETY_RISK_LEVELS,
    PUBLIC_IMPACT_LEVELS,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PRIORITY_HIGH,
    PRIORITY_CRITICAL
)

def calculate_priority_score(
    severity: str,
    safety_risk: str,
    public_impact: str,
    created_at_dt: datetime.datetime = None
) -> dict:
    """
    Computes a composite Priority Score (0 to 100) and Priority Level.
    
    Formula:
        Base Score (0 - 100) = (Severity_val * 0.35 + Safety_val * 0.35 + Impact_val * 0.30) * 25
        Aging Factor (0 - 20) = min(20.0, days_elapsed * 2.0)
        Total Priority = min(100, round(Base Score + Aging Factor))
        
    Priority Thresholds:
        - 0 to 39: Low
        - 40 to 64: Medium
        - 65 to 84: High
        - 85 to 100: Critical
    """
    # 1. Resolve numeric scale values (1 to 4)
    sev_val = SEVERITY_LEVELS.get(severity, 2)
    risk_val = SAFETY_RISK_LEVELS.get(safety_risk, 2)
    impact_val = PUBLIC_IMPACT_LEVELS.get(public_impact, 2)

    # 2. Weighted Base Score (scales 1..4 weighted sum to 25..100)
    weighted_sum = (
        (sev_val * SEVERITY_WEIGHT) +
        (risk_val * SAFETY_RISK_WEIGHT) +
        (impact_val * PUBLIC_IMPACT_WEIGHT)
    )
    base_score = weighted_sum * 25.0

    # 3. Dynamic Waiting Time Escalation (Aging Bonus)
    aging_bonus = 0.0
    days_waiting = 0.0
    if created_at_dt is not None:
        now = datetime.datetime.now()
        if isinstance(created_at_dt, str):
            try:
                created_at_dt = datetime.datetime.strptime(created_at_dt, "%Y-%m-%d %H:%M:%S")
            except Exception:
                created_at_dt = now

        delta_seconds = max(0, (now - created_at_dt).total_seconds())
        days_waiting = delta_seconds / 86400.0
        # Add points per day waiting (capped at 20 points)
        aging_bonus = min(20.0, days_waiting * AGING_BONUS_PER_DAY)

    # 4. Total Score Calculation
    total_score = min(100, int(round(base_score + aging_bonus)))

    # 5. Determine Priority Category
    if total_score >= 85:
        priority_level = PRIORITY_CRITICAL
    elif total_score >= 65:
        priority_level = PRIORITY_HIGH
    elif total_score >= 40:
        priority_level = PRIORITY_MEDIUM
    else:
        priority_level = PRIORITY_LOW

    return {
        "priority_score": total_score,
        "priority_level": priority_level,
        "base_score": round(base_score, 1),
        "aging_bonus": round(aging_bonus, 1),
        "days_waiting": round(days_waiting, 1),
        "breakdown": {
            "severity_component": round(sev_val * SEVERITY_WEIGHT * 25.0, 1),
            "safety_component": round(risk_val * SAFETY_RISK_WEIGHT * 25.0, 1),
            "impact_component": round(impact_val * PUBLIC_IMPACT_WEIGHT * 25.0, 1)
        }
    }
