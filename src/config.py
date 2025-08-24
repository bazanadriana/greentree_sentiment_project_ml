from dataclasses import dataclass

@dataclass
class Columns:
    text: str = "text"
    date: str = "date"
    employee_id: str = "employee_id"
    department: str = "department"

POS_THRESH = 0.05
NEG_THRESH = -0.05

NEG_RATIO_W = 0.5
TREND_DROP_W = 0.3
LOW_AVG_W = 0.2
HIGH_RISK_THRESHOLD = 70.0
LOW_AVG_PIVOT = 0.2

RANK_MONTHS = 3
