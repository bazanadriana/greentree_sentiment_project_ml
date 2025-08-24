import os
import pandas as pd
from dotenv import load_dotenv
from typing import Tuple
from .config import Columns

def load_env_columns() -> Columns:
    load_dotenv(override=True)
    return Columns(
        text=os.getenv("TEXT_COLUMN", "text"),
        date=os.getenv("DATE_COLUMN", "date"),
        employee_id=os.getenv("EMPLOYEE_ID_COLUMN", "employee_id"),
        department=os.getenv("DEPARTMENT_COLUMN", "department"),
    )

def read_dataset() -> Tuple[pd.DataFrame, Columns]:
    load_dotenv(override=True)
    data_path = os.getenv("DATA_PATH", "./data/surveys.csv")
    cols = load_env_columns()
    if data_path.lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    df.columns = [c.strip() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    def pick(name: str) -> str:
        return colmap.get(name.lower(), name)
    needed = [pick(cols.text), pick(cols.date), pick(cols.employee_id)]
    for c in needed:
        if c not in df.columns:
            raise KeyError(f"Missing required column: '{c}'. Configure via .env to match your headers.")
    df[pick(cols.date)] = pd.to_datetime(df[pick(cols.date)], errors="coerce")
    return df, Columns(
        text=pick(cols.text),
        date=pick(cols.date),
        employee_id=pick(cols.employee_id),
        department=pick(cols.department) if pick(cols.department) in df.columns else cols.department,
    )

def year_month(s):
    return s.dt.to_period('M').astype(str)
