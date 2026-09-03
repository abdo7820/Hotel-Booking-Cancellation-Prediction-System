"""
train.py
=========
يتشغل مرة واحدة على Kaggle (فيه GPU/موارد كفاية). بياخد hotel_bookings.csv
ويعمل كل خطوات الـ EDA / Cleaning / Feature Engineering / Preprocessing / Training
بالظبط زي النوتبوك الأصلي، وبعدين يحفظ:

  artifacts/
    preprocessing.pkl          -> كل الـ encoders/scalers/maps المطلوبة وقت الـ inference
    models/<model_name>.pkl    -> كل موديل متدرب لوحده
    metrics.json               -> جدول مقارنة الموديلات (مرتب من الأحسن للأسوأ بالـ F1)
    feature_importance.json    -> أهم 15 Feature (من أفضل tree-based model)
    insights.json              -> كل الـ insights/الاحصائيات المهمة اللي هتتعرض في الداشبورد
    feature_schema.json        -> شكل الـ Input المطلوب في الـ /predict (الأعمدة الخام)

بعد ما تشغله على Kaggle، نزّل مجلد artifacts/ بالكامل وحطه جنب app/ عشان الـ API يقرأه.

الاستخدام على Kaggle:
    !python train.py --data /kaggle/input/<dataset>/hotel_bookings.csv --out artifacts
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
LEAD_TIME_ORDER = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "180+ days"]
MEAL_ORDER = {"SC": 0, "BB": 1, "HB": 2, "FB": 3}
DROP_COLS = [
    "arrival_date", "arrival_date_month", "arrival_date_year",
    "arrival_date_month_num", "arrival_date_day_of_month",
    "arrival_date_week_number", "total_prev_stays",
]


def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    return "Fall"


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduces the cleaning + feature engineering steps of the notebook."""
    df = df.drop(columns=["reservation_status", "reservation_status_date"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)

    df["has_company"] = df["company"].notnull().astype(int)
    df = df.drop(columns=["company"])
    imputer = SimpleImputer(strategy="most_frequent")
    df[["country"]] = imputer.fit_transform(df[["country"]])
    df["children"] = df["children"].replace(np.nan, 0)
    df["agent"] = df["agent"].replace(np.nan, 0)

    df = df.drop(df[(df["adults"] == 0) & (df["children"] == 0) & (df["babies"] == 0)].index)
    df = df[df["adr"] > 0].copy()

    Q1, Q3 = df["adr"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    df = df[df["adr"] <= upper_bound].copy()

    df["meal"] = df["meal"].replace("Undefined", "SC")
    df = df[df["market_segment"] != "Undefined"].copy()
    df = df[df["distribution_channel"] != "Undefined"].copy()
    df = df.reset_index(drop=True)

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df = df[df["total_guests"] > 0].copy()

    df["is_family"] = ((df["children"] > 0) | (df["babies"] > 0)).astype(int)
    df["room_type_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    df["total_prev_stays"] = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["is_returning_customer"] = (df["total_prev_stays"] > 0).astype(int)

    df["arrival_date_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_date_month_num"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str),
        format="%Y-%m-%d",
    )
    df["arrival_season"] = df["arrival_date_month_num"].apply(month_to_season)

    df["adr_per_person"] = df["adr"] / df["total_guests"]
    df["estimated_total_cost"] = df["adr"] * df["total_nights"]

    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 90, 180, df["lead_time"].max()],
        labels=LEAD_TIME_ORDER,
    )
    df["has_prior_cancellation"] = (df["previous_cancellations"] > 0).astype(int)

    return df, imputer


def build_insights(df: pd.DataFrame) -> dict:
    """كل الأرقام/الجداول اللي الداشبورد هيعرضها كـ Insights & Charts."""
    insights = {}
    insights["target_distribution"] = (
        df["is_canceled"].value_counts(normalize=True).mul(100).round(2).to_dict()
    )
    insights["cancel_rate_by_market_segment"] = (
        df.groupby("market_segment")["is_canceled"].mean().mul(100).round(2)
        .sort_values(ascending=False).to_dict()
    )
    insights["cancel_rate_by_season"] = (
        df.groupby("arrival_season")["is_canceled"].mean().mul(100).round(2)
        .reindex(["Winter", "Spring", "Summer", "Fall"]).to_dict()
    )
    insights["cancel_rate_by_deposit_type"] = (
        df.groupby("deposit_type")["is_canceled"].mean().mul(100).round(2)
        .sort_values(ascending=False).to_dict()
    )
    insights["cancel_rate_by_lead_time_bucket"] = (
        df.groupby("lead_time_bucket", observed=True)["is_canceled"].mean().mul(100).round(2)
        .reindex(LEAD_TIME_ORDER).to_dict()
    )
    insights["cancel_rate_room_type_changed"] = (
        df.groupby("room_type_changed")["is_canceled"].mean().mul(100).round(2).to_dict()
    )
    insights["cancel_rate_returning_customer"] = (
        df.groupby("is_returning_customer")["is_canceled"].mean().mul(100).round(2).to_dict()
    )
    insights["cancel_rate_prior_cancellation"] = (
        df.groupby("has_prior_cancellation")["is_canceled"].mean().mul(100).round(2).to_dict()
    )
    insights["hotel_distribution"] = df["hotel"].value_counts().to_dict()
    insights["meal_distribution"] = df["meal"].value_counts().to_dict()
    insights["deposit_type_distribution"] = df["deposit_type"].value_counts().to_dict()

    numeric_for_corr = ["lead_time", "adr", "total_nights", "total_guests",
                         "total_prev_stays", "is_canceled"]
    insights["correlation_matrix"] = df[numeric_for_corr].corr().round(3).to_dict()

    # Chi-square significance of key categorical features vs target
    chi_results = []
    for col in ["deposit_type", "customer_type", "market_segment",
                "distribution_channel", "hotel", "meal",
                "assigned_room_type", "reserved_room_type"]:
        contingency = pd.crosstab(df[col], df["is_canceled"])
        chi2, p, dof, _ = chi2_contingency(contingency)
        chi_results.append({
            "feature": col,
            "chi2": round(float(chi2), 2),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        })
    insights["chi_square_results"] = chi_results

    insights["summary_stats"] = {
        "total_bookings": int(len(df)),
        "overall_cancellation_rate": round(float(df["is_canceled"].mean() * 100), 2),
        "avg_lead_time": round(float(df["lead_time"].mean()), 1),
        "avg_adr": round(float(df["adr"].mean()), 2),
        "avg_total_nights": round(float(df["total_nights"].mean()), 2),
    }
    return insights


def preprocess_for_model(X_train, X_test, y_train):
    """كل خطوات التحويل قبل التدريب (Frequency Encoding / Power Transform /
    Ordinal / Scaling / One-Hot / SMOTETomek), وبيرجع كمان الـ artifacts
    اللازمة عشان نعمل نفس التحويل وقت الـ inference."""

    artifacts = {}

    agent_freq_map = X_train["agent"].value_counts(normalize=True)
    X_train["agent_freq"] = X_train["agent"].map(agent_freq_map)
    X_test["agent_freq"] = X_test["agent"].map(agent_freq_map).fillna(0)
    country_freq_map = X_train["country"].value_counts(normalize=True)
    X_train["country_freq"] = X_train["country"].map(country_freq_map)
    X_test["country_freq"] = X_test["country"].map(country_freq_map).fillna(0)
    X_train = X_train.drop(columns=["agent", "country"])
    X_test = X_test.drop(columns=["agent", "country"])
    artifacts["agent_freq_map"] = agent_freq_map.to_dict()
    artifacts["country_freq_map"] = country_freq_map.to_dict()

    numeric = X_train.select_dtypes(include="number").columns
    pt = PowerTransformer(method="yeo-johnson")
    X_train[numeric] = pd.DataFrame(pt.fit_transform(X_train[numeric]), columns=numeric, index=X_train.index)
    X_test[numeric] = pd.DataFrame(pt.transform(X_test[numeric]), columns=numeric, index=X_test.index)
    artifacts["power_transformer"] = pt
    artifacts["power_transformer_cols"] = list(numeric)

    oe = OrdinalEncoder(categories=[LEAD_TIME_ORDER])
    X_train[["lead_time_bucket"]] = oe.fit_transform(X_train[["lead_time_bucket"]].astype(str))
    X_test[["lead_time_bucket"]] = oe.transform(X_test[["lead_time_bucket"]].astype(str))
    artifacts["lead_time_encoder"] = oe

    X_train["meal"] = X_train["meal"].map(MEAL_ORDER)
    X_test["meal"] = X_test["meal"].map(MEAL_ORDER)

    numeric = X_train.select_dtypes(include="number").columns
    sc = StandardScaler()
    X_train[numeric] = pd.DataFrame(sc.fit_transform(X_train[numeric]), columns=numeric, index=X_train.index)
    X_test[numeric] = pd.DataFrame(sc.transform(X_test[numeric]), columns=numeric, index=X_test.index)
    artifacts["scaler"] = sc
    artifacts["scaler_cols"] = list(numeric)

    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    categorical = X_train.select_dtypes(include="object").columns
    encoded_train = ohe.fit_transform(X_train[categorical])
    encoded_train_df = pd.DataFrame(encoded_train, columns=ohe.get_feature_names_out(categorical), index=X_train.index)
    X_train = pd.concat([X_train.drop(columns=categorical), encoded_train_df], axis=1)
    encoded_test = ohe.transform(X_test[categorical])
    encoded_test_df = pd.DataFrame(encoded_test, columns=ohe.get_feature_names_out(categorical), index=X_test.index)
    X_test = pd.concat([X_test.drop(columns=categorical), encoded_test_df], axis=1)
    artifacts["ohe"] = ohe
    artifacts["ohe_cols"] = list(categorical)

    artifacts["final_columns"] = list(X_train.columns)

    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, artifacts


def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1, tree_method="hist"),
    }

    trained_models, results = {}, []
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        results.append(_score_row(name, y_test, y_pred, model, X_test))

    # XGBoost tuning (F1)
    print("Tuning XGBoost (F1) ...")
    tuning_model = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1, tree_method="hist")
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    search_f1 = RandomizedSearchCV(tuning_model, param_distributions=param_dist, n_iter=30,
                                    scoring="f1", cv=3, random_state=42, verbose=0)
    search_f1.fit(X_train, y_train)
    best_f1_model = search_f1.best_estimator_
    y_pred = best_f1_model.predict(X_test)
    trained_models["XGBoost Tuned for F1"] = best_f1_model
    results.append(_score_row("XGBoost Tuned for F1", y_test, y_pred, best_f1_model, X_test))

    print("Tuning XGBoost (Accuracy) ...")
    search_acc = RandomizedSearchCV(tuning_model, param_distributions=param_dist, n_iter=30,
                                     scoring="accuracy", cv=3, random_state=42, verbose=0)
    search_acc.fit(X_train, y_train)
    best_acc_model = search_acc.best_estimator_
    y_pred = best_acc_model.predict(X_test)
    trained_models["XGBoost Tuned for Accuracy"] = best_acc_model
    results.append(_score_row("XGBoost Tuned for Accuracy", y_test, y_pred, best_acc_model, X_test))

    # Ensembles
    print("Training Voting & Stacking ensembles ...")
    estimators = [
        ("tuned_boost", best_acc_model),
        ("random_forest", trained_models["Random Forest"]),
        ("gradient_boost", trained_models["Gradient Boosting"]),
    ]
    voting_clf = VotingClassifier(estimators=estimators, voting="soft")
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)
    trained_models["Soft Voting"] = voting_clf
    results.append(_score_row("Soft Voting", y_test, y_pred, voting_clf, X_test))

    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
    )
    stacking_clf.fit(X_train, y_train)
    y_pred = stacking_clf.predict(X_test)
    trained_models["Stacking"] = stacking_clf
    results.append(_score_row("Stacking", y_test, y_pred, stacking_clf, X_test))

    return trained_models, results


def _score_row(name, y_test, y_pred, model, X_test):
    row = {
        "Model": name,
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1": round(float(f1_score(y_test, y_pred)), 4),
    }
    try:
        proba = model.predict_proba(X_test)[:, 1]
        row["ROC_AUC"] = round(float(roc_auc_score(y_test, proba)), 4)
    except Exception:
        row["ROC_AUC"] = None
    return row


def main(data_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    print("Loading data ...")
    df_raw = pd.read_csv(data_path)

    print("Cleaning & feature engineering ...")
    df, country_imputer = clean_and_engineer(df_raw)

    print("Building insights ...")
    insights = build_insights(df)
    with open(os.path.join(out_dir, "insights.json"), "w") as f:
        json.dump(insights, f, indent=2, default=str)

    model_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    X = model_df.drop("is_canceled", axis=1)
    y = model_df["is_canceled"]

    # schema of RAW input the API must accept (before any encoding)
    raw_schema = {col: str(X[col].dtype) for col in X.columns}
    with open(os.path.join(out_dir, "feature_schema.json"), "w") as f:
        json.dump(raw_schema, f, indent=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Preprocessing (encoders/scalers/SMOTETomek) ...")
    X_train_p, X_test_p, y_train_p, prep_artifacts = preprocess_for_model(
        X_train.copy(), X_test.copy(), y_train.copy()
    )
    prep_artifacts["country_imputer"] = country_imputer
    joblib.dump(prep_artifacts, os.path.join(out_dir, "preprocessing.pkl"))

    print("Training models ...")
    trained_models, results = train_all_models(X_train_p, y_train_p, X_test_p, y_test)

    for name, model in trained_models.items():
        safe_name = name.replace(" ", "_").lower()
        joblib.dump(model, os.path.join(out_dir, "models", f"{safe_name}.pkl"))

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df["file"] = results_df["Model"].apply(lambda n: n.replace(" ", "_").lower() + ".pkl")
    results_df.to_json(os.path.join(out_dir, "metrics.json"), orient="records", indent=2)

    # feature importance from the best tree-based model available
    tree_model_name = None
    for candidate in results_df["Model"]:
        if hasattr(trained_models[candidate], "feature_importances_"):
            tree_model_name = candidate
            break
    if tree_model_name:
        importances = pd.Series(
            trained_models[tree_model_name].feature_importances_, index=X_train_p.columns
        )
        top_features = importances.sort_values(ascending=False).head(15)
        feature_importance = {
            "source_model": tree_model_name,
            "features": [{"feature": k, "importance": round(float(v), 4)} for k, v in top_features.items()],
        }
        with open(os.path.join(out_dir, "feature_importance.json"), "w") as f:
            json.dump(feature_importance, f, indent=2)

    print(f"\nDone. Best model: {results_df.iloc[0]['Model']} (F1={results_df.iloc[0]['F1']})")
    print(f"All artifacts saved under: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to hotel_bookings.csv")
    parser.add_argument("--out", default="artifacts", help="Output artifacts directory")
    args = parser.parse_args()
    main(args.data, args.out)
