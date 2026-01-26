

# =====================================================
# FASTAPI + ML COLLEGE RECOMMENDATION BACKEND (FINAL)
# =====================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("colleges.csv", engine="python")

# Ensure Year column exists
if "Year" not in df.columns:
    df["Year"] = 2025

df["cutoff_raw"] = df["cutoff"]

df["cutoff"] = (
    df["cutoff"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)")[0]
    .astype(float)
)

df = df.dropna(subset=["cutoff"])

# -------------------------------
# DETECT CUTOFF TYPE
# -------------------------------
def detect_cutoff_nature(raw):
    raw = str(raw).lower()
    if "air" in raw or "crl" in raw:
        return "rank"
    if "%" in raw or "percentile" in raw or "ile" in raw:
        return "percentile"
    if "mark" in raw:
        return "marks"
    return "percentage"

def detect_exam_hint(raw):
    raw = str(raw).lower()
    if "gate" in raw:
        return "gate"
    if "jee" in raw or "air" in raw or "crl" in raw:
        return "jee"
    return "other"

df["cutoff_nature"] = df["cutoff_raw"].apply(detect_cutoff_nature)
df["exam_hint"] = df["cutoff_raw"].apply(detect_exam_hint)

# -------------------------------
# NORMALIZE TEXT
# -------------------------------
text_cols = ["field", "branches", "category", "location"]

for col in text_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# -------------------------------
# ENCODE TEXT
# -------------------------------
encoders = {}

for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -------------------------------
# TRAIN ML MODEL
# -------------------------------
df["college_score"] = 1 / (df["cutoff"] + 1)

X = df[["field", "branches", "cutoff", "category", "location", "Year"]]
y = df["college_score"]

model = RandomForestRegressor(
    n_estimators=80,
    random_state=42
)

model.fit(X, y)

# -------------------------------
# FIELD → BRANCH API
# -------------------------------
@app.get("/branches")
def get_branches(field: str):
    field = field.lower().strip()

    if field not in encoders["field"].classes_:
        return []

    field_e = encoders["field"].transform([field])[0]
    data = df[df["field"] == field_e]

    branches = encoders["branches"].inverse_transform(
        data["branches"].unique()
    )

    return sorted(branches.tolist())

# -------------------------------
# INPUT SCHEMA
# -------------------------------
class UserInput(BaseModel):
    field: str
    category: str
    cutoff_value: float
    exam_mode: str
    branches: Optional[List[str]] = None
    page: int = 1
    limit: int = 10

# -------------------------------
# RECOMMENDATION API (FINAL)
# -------------------------------
@app.post("/recommend")
def recommend(data: UserInput):

    field = data.field.lower().strip()
    exam_mode = data.exam_mode.lower().strip()

    # Auto handle Junior College
    if field == "junior college":
        exam_mode = "percentage"

    if exam_mode in ["arts", "science", "commerce", "junior", "junior_college", "diploma", "percentage"]:
        exam_mode = "percentage"

    if exam_mode in ["ba_llb", "bba_llb", "bcom_llb", "bsc_llb"]:
        exam_mode = "llb"

    if field not in encoders["field"].classes_:
        return {"total_colleges": 0, "results": []}

    field_e = encoders["field"].transform([field])[0]

    category = data.category.lower().strip()
    if category not in encoders["category"].classes_:
        return {"total_colleges": 0, "results": []}
    category_e = encoders["category"].transform([category])[0]

    if exam_mode == "jee":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"] == "rank") &
            (df["cutoff"] >= data.cutoff_value)
        ]

    elif exam_mode == "mhtcet":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"] == "percentile") &
            (df["cutoff"] <= data.cutoff_value)
        ]

    elif exam_mode == "medical":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"] == "marks") &
            (df["cutoff"] <= data.cutoff_value)
        ]

    elif exam_mode == "mba":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"].isin(["percentile", "percentage"])) &
            (df["cutoff"] <= data.cutoff_value)
        ]

    elif exam_mode == "gate":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["exam_hint"] == "gate") &
            (df["cutoff"] <= data.cutoff_value)
        ]

    elif exam_mode == "llb":
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"].isin(["percentile", "percentage"])) &
            (df["cutoff"] <= data.cutoff_value)
        ]

    else:
        eligible = df[
            (df["field"] == field_e) &
            (df["category"] == category_e) &
            (df["cutoff_nature"].isin(["percentage", "percentile"])) &
            (df["cutoff"] <= data.cutoff_value)
        ]

    # -------- OPTIONAL BRANCH FILTER (MAX 3) --------
    if data.branches:
        if len(data.branches) > 3:
            return {"error": "You can select maximum 3 branches only"}

        encoded_branches = []
        for b in data.branches:
            b = b.lower().strip()
            if b in encoders["branches"].classes_:
                encoded_branches.append(encoders["branches"].transform([b])[0])

        if encoded_branches:
            eligible = eligible[eligible["branches"].isin(encoded_branches)]

    if eligible.empty:
        return {"total_colleges": 0, "results": []}

    eligible = eligible.copy()
    eligible["ml_score"] = model.predict(
        eligible[["field", "branches", "cutoff", "category", "location", "Year"]]
    )

    eligible = eligible.sort_values("ml_score", ascending=True)

    page = max(data.page, 1)
    limit = max(data.limit, 1)

    start = (page - 1) * limit
    end = start + limit

    paginated = eligible.iloc[start:end].copy()

    paginated["branches"] = encoders["branches"].inverse_transform(paginated["branches"])
    paginated["location"] = encoders["location"].inverse_transform(paginated["location"])

    return {
        "total_colleges": len(eligible),
        "page": page,
        "limit": limit,
        "results": paginated[
            ["name", "branches", "cutoff_raw", "location", "Year"]
        ].to_dict(orient="records")
    }