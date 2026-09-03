"""
FastAPI backend لمشروع Hotel Booking Cancellation Prediction.

بيقرأ كل الـ artifacts اللي train.py حفظها (models/, preprocessing.pkl,
metrics.json, insights.json, feature_importance.json, feature_schema.json)
وبيعرضها + بيعمل predict باستخدام أي موديل يختاره المستخدم.
"""

import json
import os
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, Field

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
STATIC_DIR = os.environ.get("STATIC_DIR", "static")

LEAD_TIME_ORDER = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"]
MEAL_ORDER = {"SC": 0, "BB": 1, "HB": 2, "FB": 3}
DROP_COLS = [
    "arrival_date", "arrival_date_month", "arrival_date_year",
    "arrival_date_month_num", "arrival_date_day_of_month",
    "arrival_date_week_number", "total_prev_stays",
]

app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    description="API لتوقع إلغاء حجوزات الفنادق، بيدعم اختيار الموديل من مجموعة موديلات متدربة.",
    version="1.0.0",
)

_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: Dict[str, Any] = {
    "preprocessing": None,
    "models": {},
    "metrics": [],
    "insights": {},
    "feature_importance": {},
    "feature_schema": {},
}


def _load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


@app.on_event("startup")
def load_artifacts():
    prep_path = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
    if os.path.exists(prep_path):
        _state["preprocessing"] = joblib.load(prep_path)
    else:
        print(f"[WARN] {prep_path} not found. /predict will not work until artifacts are added.")

    _state["metrics"] = _load_json(os.path.join(ARTIFACTS_DIR, "metrics.json"), [])
    _state["insights"] = _load_json(os.path.join(ARTIFACTS_DIR, "insights.json"), {})
    _state["feature_importance"] = _load_json(os.path.join(ARTIFACTS_DIR, "feature_importance.json"), {})
    _state["feature_schema"] = _load_json(os.path.join(ARTIFACTS_DIR, "feature_schema.json"), {})

    if os.path.isdir(MODELS_DIR):
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(".pkl"):
                model_key = fname[:-4]
                _state["models"][model_key] = joblib.load(os.path.join(MODELS_DIR, fname))
        print(f"Loaded {len(_state['models'])} models: {list(_state['models'].keys())}")
    else:
        print(f"[WARN] {MODELS_DIR} not found. Run train.py first and copy artifacts/ here.")

    # لو موديل اتشال من artifacts/models (زي موديلات كبيرة الحجم) بس لسه
    # مذكور في metrics.json، منشيله من القايمة تلقائيًا عشان الداشبورد
    # ما يعرضش موديلات مش موجودة فعليًا أو يسمح بـ /predict عليها.
    if _state["metrics"]:
        available_keys = set(_state["models"].keys())
        _state["metrics"] = [
            row for row in _state["metrics"]
            if row.get("file", "").replace(".pkl", "") in available_keys
        ]


class PredictRequest(BaseModel):
    model_key: str = Field(..., description="اسم ملف الموديل زي ما ظاهر في /models (مثال: xgboost, random_forest)")
    booking: Dict[str, Any] = Field(..., description="بيانات الحجز الخام (نفس أعمدة feature_schema)")


class PredictResponse(BaseModel):
    model_used: str
    prediction: int
    prediction_label: str
    cancellation_probability: Optional[float] = None


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": list(_state["models"].keys()),
        "preprocessing_loaded": _state["preprocessing"] is not None,
    }


@app.get("/models")
def list_models():
    """كل الموديلات مرتبة من الأحسن للأسوأ حسب F1، مع اسم الملف المطلوب في /predict."""
    if not _state["metrics"]:
        raise HTTPException(status_code=404, detail="metrics.json غير موجود. شغّل train.py الأول.")
    return _state["metrics"]


@app.get("/features")
def feature_importance():
    if not _state["feature_importance"]:
        raise HTTPException(status_code=404, detail="feature_importance.json غير موجود.")
    return _state["feature_importance"]


@app.get("/insights")
def insights():
    if not _state["insights"]:
        raise HTTPException(status_code=404, detail="insights.json غير موجود.")
    return _state["insights"]


@app.get("/schema")
def schema():
    if not _state["feature_schema"]:
        raise HTTPException(status_code=404, detail="feature_schema.json غير موجود.")
    return _state["feature_schema"]


def _month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    return "Fall"


def _engineer_single_booking(raw: Dict[str, Any]) -> pd.DataFrame:
    """بياخد dict خام (نفس أعمدة hotel_bookings.csv الأساسية) ويعمل نفس
    Feature Engineering اللي في train.py عشان يوصل لنفس الأعمدة اللي الموديل اتدرب عليها."""
    df = pd.DataFrame([raw])

    df["has_company"] = 1 if raw.get("company") not in (None, "", 0) else 0
    if "company" in df.columns:
        df = df.drop(columns=["company"])

    df["children"] = df.get("children", 0)
    df["agent"] = df.get("agent", 0)

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["room_type_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["total_prev_stays"] = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["is_returning_customer"] = (df["total_prev_stays"] > 0).astype(int)

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    month_num = month_map.get(raw.get("arrival_date_month"), 1)
    df["arrival_season"] = _month_to_season(month_num)

    df["adr_per_person"] = df["adr"] / df["total_guests"]
    df["estimated_total_cost"] = df["adr"] * df["total_nights"]

    lead_time = raw.get("lead_time", 0)
    if lead_time <= 7:
        bucket = "0-7 days"
    elif lead_time <= 30:
        bucket = "8-30 days"
    elif lead_time <= 90:
        bucket = "31-90 days"
    elif lead_time <= 180:
        bucket = "91-180 days"
    else:
        bucket = "180+ days"
    df["lead_time_bucket"] = bucket
    df["has_prior_cancellation"] = (df["previous_cancellations"] > 0).astype(int)

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df


def _apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    prep = _state["preprocessing"]
    if prep is None:
        raise HTTPException(status_code=503, detail="preprocessing.pkl غير موجود. ضيف artifacts/ الأول.")

    agent_freq_map = prep["agent_freq_map"]
    country_freq_map = prep["country_freq_map"]
    df["agent_freq"] = df["agent"].map(agent_freq_map).fillna(0)
    df["country_freq"] = df["country"].map(country_freq_map).fillna(0)
    df = df.drop(columns=["agent", "country"])

    pt = prep["power_transformer"]
    pt_cols = prep["power_transformer_cols"]
    df[pt_cols] = pt.transform(df[pt_cols])

    oe = prep["lead_time_encoder"]
    df[["lead_time_bucket"]] = oe.transform(df[["lead_time_bucket"]].astype(str))

    df["meal"] = df["meal"].map(MEAL_ORDER)

    sc = prep["scaler"]
    sc_cols = prep["scaler_cols"]
    df[sc_cols] = sc.transform(df[sc_cols])

    ohe = prep["ohe"]
    ohe_cols = prep["ohe_cols"]
    encoded = ohe.transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    final_cols = prep["final_columns"]
    for col in final_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[final_cols]

    return df


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_key not in _state["models"]:
        raise HTTPException(
            status_code=404,
            detail=f"موديل '{req.model_key}' مش موجود. الموديلات المتاحة: {list(_state['models'].keys())}",
        )

    try:
        engineered = _engineer_single_booking(req.booking)
        processed = _apply_preprocessing(engineered)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"عمود ناقص في بيانات الحجز: {e}")

    model = _state["models"][req.model_key]
    pred = int(model.predict(processed)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(processed)[0][1]), 4)

    return PredictResponse(
        model_used=req.model_key,
        prediction=pred,
        prediction_label="Canceled" if pred == 1 else "Not Canceled",
        cancellation_probability=proba,
    )


if os.path.isdir(STATIC_DIR):
    app.mount("/dashboard", StaticFiles(directory=STATIC_DIR, html=True), name="dashboard")