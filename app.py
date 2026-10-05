from __future__ import annotations

import html
import itertools
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------------------
# 1. CONFIGURATION  (everything you may want to tweak is here)
# ----------------------------------------------------------------------------------------
CSV_NAME = "fraud_dataset.csv"
CSV_PATH = Path(__file__).resolve().parent / CSV_NAME
TARGET = "is_fraud"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_TREES = 300

# Risk thresholds (fraud risk score in %). Easy to change.
LOW_RISK_MAX = 30.0    # below this  -> LOW
HIGH_RISK_MIN = 70.0   # above this  -> HIGH   (in between -> MEDIUM)

# Indicator table: observed dataset fraud-rate cut-offs
INDICATOR_WARN = 0.50
INDICATOR_WATCH = 0.25

MAX_ALLOWED_AMOUNT = 1_000_000.0          # sanity limit for the amount field
AMOUNT_EDGES = [2000, 5000, 10000]        # bands used only for the indicator table
AMOUNT_BAND_LABELS = ["Under ₹2,000", "₹2,000 – ₹5,000", "₹5,000 – ₹10,000", "Above ₹10,000"]

# ---- Columns of the CSV that the deployable model is built from -------------------------
# Raw CSV columns that must exist (verified at start-up)
REQUIRED_RAW_COLUMNS = [
    TARGET, "amount", "transaction_time_of_day", "request_description_keywords",
    "recognized_screen_sharing_apps", "transaction_type", "merchant_category_code",
    "session_source", "authorization_method", "handle_typo_analysis",
    "handle_registration_pattern",
]

# Engineered model features (exact order the model is trained on and receives at prediction)
NUMERIC_FEATURES = ["amount", "hour_of_day", "urgent_language", "screen_sharing_requested"]
CATEGORICAL_FEATURES = [
    "transaction_type", "merchant_category_code", "session_source",
    "authorization_method", "handle_typo_analysis", "handle_registration_pattern",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fields where the customer may answer "Not sure" (handled by averaging over dataset base rates)
UNKNOWN_CAPABLE = ["handle_typo_analysis", "handle_registration_pattern"]

# Customer-friendly wording for dataset values (unseen values fall back to a title-cased label)
VALUE_LABELS = {
    "transaction_type": {
        "payment": "Payment - I am sending money",
        "collection_request": "Collect request - someone is asking me for money",
    },
    "merchant_category_code": {
        "food": "Food & dining", "retail": "Retail & shopping", "services": "Services",
        "utilities": "Utilities & bills", "entertainment": "Entertainment",
        "unknown": "Unknown / not sure",
    },
    "session_source": {
        "app": "I opened it myself in my UPI app",
        "link": "I tapped a link sent to me",
    },
    "authorization_method": {
        "pin": "UPI PIN in my UPI app (standard)",
        "otp": "An OTP is being requested",
    },
}
FEATURE_LABELS = {
    "amount": "Transaction amount",
    "hour_of_day": "Time of day",
    "urgent_language": "Urgent / pressuring request wording",
    "screen_sharing_requested": "Screen-sharing app requested",
    "transaction_type": "Transaction type",
    "merchant_category_code": "Merchant category",
    "session_source": "How the payment was opened",
    "authorization_method": "Approval method (PIN / OTP)",
    "handle_typo_analysis": "Lookalike / misspelled UPI handle",
    "handle_registration_pattern": "Recently created UPI handle",
}

# Columns in the CSV that look like identifiers / free text (never used as features)
IDENTIFIER_LIKE = [
    "transaction_id", "user_id", "merchant_id", "device_id", "ip_address", "location",
    "description", "url_referrer", "business_name_match", "timestamp",
]

RISK_LEVELS = {
    "LOW": dict(
        icon="🟢", label="LOW RISK", color="#22c55e", bg="linear-gradient(135deg,#052e1a,#0b3d26)",
        headline="Transaction appears relatively low risk.",
        prediction="Likely Genuine",
        recommendation="Review the recipient and transaction details before proceeding. "
                       "No major fraud indicators were detected by the model.",
    ),
    "MEDIUM": dict(
        icon="🟠", label="MEDIUM RISK", color="#f59e0b", bg="linear-gradient(135deg,#3b2506,#4a2f08)",
        headline="Transaction may require additional verification.",
        prediction="Suspicious Transaction",
        recommendation="Verify the receiver's UPI ID, identity and payment purpose before proceeding.",
    ),
    "HIGH": dict(
        icon="🔴", label="HIGH RISK", color="#ef4444", bg="linear-gradient(135deg,#3b0a0a,#4c1010)",
        headline="Potential fraud detected.",
        prediction="Potentially Fraudulent",
        recommendation="Do not proceed until you independently verify the recipient and payment request.",
    ),
}

PAGE_HOME = "🏠 Home"
PAGE_CHECK = "🔍 Check Transaction"
PAGE_INSIGHTS = "📊 Model Insights"
PAGE_ABOUT = "ℹ️ About"
PAGES = [PAGE_HOME, PAGE_CHECK, PAGE_INSIGHTS, PAGE_ABOUT]

PRIVACY_NOTE = (
    "This demonstration tool analyzes transaction information entered by the user. Do not enter "
    "passwords, UPI PINs, OTPs, bank passwords, card PINs, or other authentication credentials."
)
MODEL_DISCLAIMER = (
    "This tool provides a machine-learning based risk assessment and does not guarantee that a "
    "transaction is safe or fraudulent."
)


# ----------------------------------------------------------------------------------------
# 2. SMALL HELPERS
# ----------------------------------------------------------------------------------------
def esc(text) -> str:
    """HTML-escape any user supplied text before it is placed in custom HTML."""
    return html.escape(str(text), quote=True)


def html_block(markup: str) -> str:
    """Collapse indentation so Markdown never treats custom HTML as a code block."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def show_html(markup: str) -> None:
    st.markdown(html_block(markup), unsafe_allow_html=True)


def format_inr(value: float) -> str:
    """Format a number as Indian-rupee currency with Indian digit grouping (e.g. ₹1,25,000)."""
    whole, _, dec = f"{abs(float(value)):.2f}".partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        whole = f"{head},{tail}"
    sign = "-" if value < 0 else ""
    return f"₹{sign}{whole}" + ("" if dec == "00" else f".{dec}")


def value_label(feature: str, value: str) -> str:
    return VALUE_LABELS.get(feature, {}).get(value, str(value).replace("_", " ").title())


def risk_level(score: float) -> str:
    if score < LOW_RISK_MAX:
        return "LOW"
    if score > HIGH_RISK_MIN:
        return "HIGH"
    return "MEDIUM"


# ----------------------------------------------------------------------------------------
# 3. DATA LOADING + VALIDATION
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading fraud_dataset.csv …")
def load_dataset(path_str: str, signature: float):
    """Load the CSV once. Returns (dataframe, error_message). `signature` = file mtime."""
    path = Path(path_str)
    if not path.exists():
        return None, f"{CSV_NAME} was not found. Place it in the same folder as app.py."
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable / corrupt CSV
        return None, f"{CSV_NAME} could not be read ({type(exc).__name__}). Please check the file."

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        return None, "Your CSV is missing required columns: " + ", ".join(missing)
    if df.empty:
        return None, f"{CSV_NAME} contains no rows."
    if not set(df[TARGET].dropna().unique()) <= {0, 1} or df[TARGET].isna().any():
        return None, f"The target column '{TARGET}' must contain only 0 (genuine) and 1 (fraud)."
    if df[TARGET].nunique() < 2:
        return None, f"The target column '{TARGET}' needs both genuine and fraud examples."

    # The form's Yes/No answers are mapped to these dataset values - make sure they exist.
    expected = {
        "handle_typo_analysis": {"none", "typo_squatting"},
        "handle_registration_pattern": {"none", "recent"},
    }
    for col, vals in expected.items():
        if not vals <= set(df[col].dropna().astype(str).unique()):
            return None, f"Column '{col}' does not contain the expected values {sorted(vals)}."
    return df, None


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Raw CSV -> model features. This ONE function defines the feature structure used for training.
    (User input is built directly in the same FEATURE_COLUMNS layout at prediction time.)
    """
    out = pd.DataFrame(index=raw.index)
    out["amount"] = pd.to_numeric(raw["amount"], errors="coerce")
    out["hour_of_day"] = pd.to_numeric(raw["transaction_time_of_day"], errors="coerce")
    # '[]' in the CSV means "no keywords" / "no apps"; anything else means present
    out["urgent_language"] = (
        raw["request_description_keywords"].fillna("[]").astype(str).str.strip() != "[]"
    ).astype(int)
    out["screen_sharing_requested"] = (
        raw["recognized_screen_sharing_apps"].fillna("[]").astype(str).str.strip() != "[]"
    ).astype(int)
    for col in CATEGORICAL_FEATURES:
        out[col] = raw[col].astype(str)
    return out[FEATURE_COLUMNS]


# ----------------------------------------------------------------------------------------
# 4. MODEL TRAINING (cached - trained once, reused on every click)
# ----------------------------------------------------------------------------------------
def make_model(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """Preprocessing + Random Forest in ONE pipeline, so training and prediction match exactly."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipe, categorical_cols))
    forest = RandomForestClassifier(
        n_estimators=N_TREES, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    return Pipeline([("prep", ColumnTransformer(transformers)), ("rf", forest)])


def score_model(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "cm": confusion_matrix(y_test, pred, labels=[0, 1]),
    }


def aggregate_importances(model: Pipeline) -> list[tuple[str, float]]:
    """Random Forest importances are per one-hot column; sum them back to the original fields."""
    names = model.named_steps["prep"].get_feature_names_out()
    importances = model.named_steps["rf"].feature_importances_
    totals = {f: 0.0 for f in FEATURE_COLUMNS}
    for name, imp in zip(names, importances):
        base = name.split("__", 1)[1]
        owners = [f for f in FEATURE_COLUMNS if base == f or base.startswith(f + "_")]
        if owners:
            totals[max(owners, key=len)] += float(imp)
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def leakage_scan(df: pd.DataFrame, max_unique: int = 12) -> pd.DataFrame:
    """
    For every low-cardinality column: how well does 'predict the majority class of each value'
    reproduce is_fraud? 1.00 means the column on its own reveals the label (target leakage).
    """
    y = df[TARGET]
    rows = []
    for col in df.columns:
        if col == TARGET:
            continue
        s = df[col].fillna("NaN").astype(str)
        if not 2 <= s.nunique() <= max_unique:
            continue
        purity = pd.crosstab(s, y).max(axis=1).sum() / len(df)
        rows.append({"column": col, "purity": float(purity)})
    return pd.DataFrame(rows, columns=["column", "purity"]).sort_values("purity", ascending=False)


def describe_columns(df: pd.DataFrame, leaky: list[str]) -> list[dict]:
    """Group the CSV's columns by how the app treats them (computed from the real file)."""
    cols = [c for c in df.columns if c != TARGET]
    used_sources = [
        "amount", "transaction_time_of_day", "request_description_keywords",
        "recognized_screen_sharing_apps",
    ] + CATEGORICAL_FEATURES
    identifiers = [c for c in IDENTIFIER_LIKE if c in cols]
    constants = [c for c in cols if df[c].dropna().nunique() <= 1]
    leaky = [c for c in leaky if c in cols and c not in used_sources]
    taken = set(used_sources) | set(identifiers) | set(constants) | set(leaky)
    backend = [c for c in cols if c not in taken]
    return [
        {"group": "Used by the model (customer can provide)", "columns": used_sources,
         "reason": "Realistic information a customer knows before paying (some are turned into simple "
                   "yes/no or hour-of-day features)."},
        {"group": "Target", "columns": [TARGET], "reason": "The label the model learns to predict. Never an input."},
        {"group": "Identifiers / free text / raw timestamp", "columns": identifiers,
         "reason": "Unique IDs, IPs, coordinates and free text do not generalise. 'timestamp' only holds "
                   "minutes:seconds (no date), so the hour comes from 'transaction_time_of_day'."},
        {"group": "Constant columns", "columns": constants,
         "reason": "Same value in every row, so they carry no information (e.g. relationship_to_requester, "
                   "upi_handle_age, social_media_presence)."},
        {"group": "Excluded - perfect label proxy (leakage)", "columns": leaky,
         "reason": "Each value belongs to exactly one class, so the column alone reveals is_fraud. "
                   "Using it would make the model a lookup table."},
        {"group": "Backend / device / behavioural signals", "columns": backend,
         "reason": "Collected by the payment platform (device, IP, typing speed, PIN entry, account history …). "
                   "A customer cannot enter them, so the deployable model does not use them."},
    ]


@st.cache_resource(show_spinner="Training the Random Forest on your fraud_dataset.csv …")
def train_bundle(_df: pd.DataFrame, signature: float) -> dict:
    """Train once and keep everything the UI needs. `_df` is not hashed; `signature` triggers refresh."""
    df = _df
    X = engineer_features(df)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = make_model(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    model.fit(X_train, y_train)
    metrics = score_model(model, X_test, y_test)

    # Sanity baseline: only amount + hour (no behavioural / pattern fields)
    base_cols = ["amount", "hour_of_day"]
    baseline = make_model(base_cols, [])
    baseline.fit(X_train[base_cols], y_train)
    baseline_metrics = score_model(baseline, X_test[base_cols], y_test)

    scan = leakage_scan(df)
    leaky = scan.loc[scan["purity"] >= 0.999, "column"].tolist()

    # Observed (descriptive) fraud rates per answer - used for the indicator table
    obs: dict[str, dict] = {}
    for col in CATEGORICAL_FEATURES + ["urgent_language", "screen_sharing_requested", "hour_of_day"]:
        grp = pd.DataFrame({"v": X[col], "y": y}).groupby("v")["y"].agg(["mean", "count"])
        obs[col] = {k: (float(r["mean"]), int(r["count"])) for k, r in grp.iterrows()}
    band = np.searchsorted(AMOUNT_EDGES, X["amount"].to_numpy(), side="right")
    grp = pd.DataFrame({"v": band, "y": y.to_numpy()}).groupby("v")["y"].agg(["mean", "count"])
    obs["amount_band"] = {int(k): (float(r["mean"]), int(r["count"])) for k, r in grp.iterrows()}

    missing_cols = df.isna().sum()
    stats = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features": int(df.shape[1] - 1),
        "fraud": int((y == 1).sum()),
        "genuine": int((y == 0).sum()),
        "fraud_pct": float(y.mean() * 100),
        "missing_total": int(missing_cols.sum()),
        "missing_cols": {c: int(n) for c, n in missing_cols[missing_cols > 0].items()},
        "amount_min": float(X["amount"].min()),
        "amount_max": float(X["amount"].max()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return {
        "model": model,
        "fraud_index": list(model.classes_).index(1),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "importances": aggregate_importances(model),
        "options": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "marginals": {c: X_train[c].value_counts(normalize=True).to_dict() for c in UNKNOWN_CAPABLE},
        "observed": obs,
        "stats": stats,
        "leak_scan": scan,
        "leaky_columns": leaky,
        "column_groups": describe_columns(df, leaky),
    }


# ----------------------------------------------------------------------------------------
# 5. PREDICTION
# ----------------------------------------------------------------------------------------
def predict_risk(bundle: dict, record: dict) -> float:
    """
    Fraud Risk Score (0-100) = P(fraud) x 100 from the trained pipeline.

    `record` holds one value per FEATURE_COLUMNS entry. For fields where the customer answered
    "Not sure" the value is None: the score is then the average over the possible values, weighted
    by how often each value occurs in the training data (no arbitrary value is invented).
    """
    unknown = [c for c in UNKNOWN_CAPABLE if record.get(c) is None]
    rows, weights = [], []
    value_lists = [list(bundle["marginals"][c].items()) for c in unknown]
    for combo in itertools.product(*value_lists) if unknown else [()]:
        row, weight = dict(record), 1.0
        for field, (val, share) in zip(unknown, combo):
            row[field] = val
            weight *= share
        rows.append(row)
        weights.append(weight)

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]            # exact training column order
    proba = bundle["model"].predict_proba(X)[:, bundle["fraud_index"]]
    return float(np.average(proba, weights=weights) * 100.0)


def build_indicators(bundle: dict, record: dict) -> list[dict]:
    """
    One row per answer: what was entered + how often that answer was fraudulent in YOUR dataset.
    This is descriptive (not a per-prediction explanation).
    """
    obs = bundle["observed"]

    def row(check, shown, rate_info, unknown=False):
        if unknown or rate_info is None:
            return {"check": check, "answer": shown, "rate": None, "n": 0, "status": "?"}
        rate, n = rate_info
        tone = "bad" if rate >= INDICATOR_WARN else ("warn" if rate >= INDICATOR_WATCH else "ok")
        status = "✓" if tone == "ok" else "⚠"
        return {"check": check, "answer": shown, "rate": rate, "n": n, "status": status, "tone": tone}

    band = int(np.searchsorted(AMOUNT_EDGES, record["amount"], side="right"))
    rows = [
        row("Transaction amount", f"{format_inr(record['amount'])} ({AMOUNT_BAND_LABELS[band]})",
            obs["amount_band"].get(band)),
        row("Time of day", f"{int(record['hour_of_day']):02d}:00 – {int(record['hour_of_day']):02d}:59",
            obs["hour_of_day"].get(int(record["hour_of_day"]))),
        row("Transaction type", value_label("transaction_type", record["transaction_type"]),
            obs["transaction_type"].get(record["transaction_type"])),
        row("Merchant category", value_label("merchant_category_code", record["merchant_category_code"]),
            obs["merchant_category_code"].get(record["merchant_category_code"])),
        row("How the payment was opened", value_label("session_source", record["session_source"]),
            obs["session_source"].get(record["session_source"])),
        row("Approval method", value_label("authorization_method", record["authorization_method"]),
            obs["authorization_method"].get(record["authorization_method"])),
        row("Urgent / pressuring wording", "Yes" if record["urgent_language"] else "No",
            obs["urgent_language"].get(int(record["urgent_language"]))),
        row("Screen-sharing app requested", "Yes" if record["screen_sharing_requested"] else "No",
            obs["screen_sharing_requested"].get(int(record["screen_sharing_requested"]))),
    ]
    for field, label, yes_value in [
        ("handle_typo_analysis", "Lookalike / misspelled UPI handle", "typo_squatting"),
        ("handle_registration_pattern", "Recently created UPI handle", "recent"),
    ]:
        val = record[field]
        if val is None:
            rows.append(row(label, "Not sure", None, unknown=True))
        else:
            rows.append(row(label, "Yes" if val == yes_value else "No", obs[field].get(val)))
    return rows


# ----------------------------------------------------------------------------------------
# 6. STYLING
# ----------------------------------------------------------------------------------------
CSS = """
<style>
#MainMenu {visibility: hidden;} footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg,#0b1220 0%,#12306b 100%); border-radius: 20px;
       padding: 2.6rem 2.4rem; color: #fff; border: 1px solid #1f2a44;}
.hero h1 {font-size: 2.6rem; margin: 0 0 .4rem 0; color: #fff; letter-spacing: -.5px;}
.hero .sub {font-size: 1.25rem; font-weight: 600; color: #7dd3fc; margin-bottom: .8rem;}
.hero .desc {font-size: 1.02rem; color: #cbd5e1; max-width: 640px;}
.card {background: #0f172a; border: 1px solid #1f2a44; border-radius: 16px; padding: 1.2rem 1.3rem;
       color: #e2e8f0; height: 100%;}
.card h4 {margin: 0 0 .35rem 0; color: #fff; font-size: 1.05rem;}
.card p {margin: 0; color: #94a3b8; font-size: .93rem; line-height: 1.45;}
.card .ico {font-size: 1.6rem; margin-bottom: .5rem;}
.section-title {font-size: 1.05rem; font-weight: 700; margin: 1.2rem 0 .2rem 0; letter-spacing: .2px;}
.section-sub {font-size: .88rem; opacity: .7; margin-bottom: .6rem;}
.steps {display: flex; gap: .5rem; flex-wrap: wrap; margin: .4rem 0 1rem 0;}
.step {background: #0f172a; color: #cbd5e1; border: 1px solid #1f2a44; border-radius: 999px;
       padding: .25rem .85rem; font-size: .82rem;}
.step b {color: #7dd3fc;}
.risk-hero {border-radius: 20px; padding: 1.8rem 2rem; color: #fff; border: 1px solid rgba(255,255,255,.12);}
.risk-label {font-size: 1rem; font-weight: 800; letter-spacing: 1.5px;}
.risk-score {font-size: 3.6rem; font-weight: 800; line-height: 1.05; margin: .2rem 0;}
.risk-score small {font-size: 1.1rem; font-weight: 600; opacity: .85; margin-left: .4rem;}
.risk-pred {font-size: 1.05rem; opacity: .95;}
.gauge {position: relative; height: 14px; border-radius: 999px; background: rgba(255,255,255,.14);
        margin: 1.2rem 0 .3rem 0; overflow: visible;}
.gauge-fill {height: 100%; border-radius: 999px;}
.gauge-tick {position: absolute; top: -4px; width: 2px; height: 22px; background: rgba(255,255,255,.55);}
.gauge-lbl {position: absolute; top: 22px; font-size: .72rem; opacity: .8; transform: translateX(-50%);}
.reco {border-radius: 14px; padding: 1.1rem 1.3rem; margin-top: 1rem; border-left: 6px solid;
       background: #0f172a; color: #e2e8f0;}
.reco .t {font-size: .8rem; letter-spacing: 1px; font-weight: 800; margin-bottom: .25rem;}
.reco .m {font-size: 1.08rem; font-weight: 600;}
table.tbl {width: 100%; border-collapse: collapse; font-size: .92rem; color: #e2e8f0;}
table.tbl th {text-align: left; color: #94a3b8; font-weight: 600; font-size: .78rem; text-transform: uppercase;
              letter-spacing: .6px; padding: .45rem .5rem; border-bottom: 1px solid #1f2a44;}
table.tbl td {padding: .55rem .5rem; border-bottom: 1px solid #172033; vertical-align: top;}
table.tbl tr:last-child td {border-bottom: none;}
.pill {display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-weight: 700; font-size: .8rem;}
.pill.ok {background: #052e1a; color: #4ade80;} .pill.warn {background: #3b2506; color: #fbbf24;}
.pill.bad {background: #3b0a0a; color: #f87171;} .pill.unk {background: #1e293b; color: #94a3b8;}
.bar-row {margin: .55rem 0;} .bar-top {display: flex; justify-content: space-between; font-size: .88rem;
         color: #e2e8f0; margin-bottom: .2rem;}
.bar-bg {height: 9px; background: #1e293b; border-radius: 999px;}
.bar-fg {height: 9px; border-radius: 999px; background: linear-gradient(90deg,#38bdf8,#3b82f6);}
table.cm {border-collapse: collapse; margin: .4rem 0; color: #e2e8f0; width: 100%;}
table.cm td, table.cm th {border: 1px solid #1f2a44; padding: .7rem .8rem; text-align: center; background: #0f172a;}
table.cm th {color: #94a3b8; font-weight: 600; font-size: .8rem;}
table.cm td.good {color: #4ade80; font-weight: 800; font-size: 1.2rem;}
table.cm td.badc {color: #f87171; font-weight: 800; font-size: 1.2rem;}
.small-note {font-size: .82rem; opacity: .75;}
section[data-testid="stSidebar"] .side-title {font-size: 1.25rem; font-weight: 800; margin-bottom: .1rem;}
section[data-testid="stSidebar"] .side-sub {font-size: .85rem; opacity: .7; margin-bottom: 1rem;}
.stButton > button, .stFormSubmitButton > button {border-radius: 10px; font-weight: 600;}
</style>
"""


# ----------------------------------------------------------------------------------------
# 7. UI COMPONENTS
# ----------------------------------------------------------------------------------------
def go_to(page: str) -> None:
    st.session_state["nav"] = page


def reset_checker() -> None:
    st.session_state["result"] = None
    st.session_state["form_id"] = st.session_state.get("form_id", 0) + 1   # fresh, empty form
    st.session_state["nav"] = PAGE_CHECK


def card(icon: str, title: str, text: str) -> str:
    return html_block(f'<div class="card"><div class="ico">{icon}</div><h4>{esc(title)}</h4><p>{esc(text)}</p></div>')


def importance_bars(bundle: dict, top_n: int = 10) -> str:
    items = bundle["importances"][:top_n]
    peak = max((v for _, v in items), default=1.0) or 1.0
    rows = ""
    for feature, value in items:
        rows += (
            '<div class="bar-row"><div class="bar-top">'
            f'<span>{esc(FEATURE_LABELS.get(feature, feature))}</span><span>{value * 100:.1f}%</span></div>'
            f'<div class="bar-bg"><div class="bar-fg" style="width:{value / peak * 100:.1f}%"></div></div></div>'
        )
    return html_block(f'<div class="card">{rows}</div>')


def indicators_table(rows: list[dict]) -> str:
    body = ""
    for r in rows:
        if r["rate"] is None:
            pill = '<span class="pill unk">? Not sure</span>'
            rate_txt = "Averaged over dataset base rates"
        else:
            cls = r["tone"]
            pill = f'<span class="pill {cls}">{r["status"]}</span>'
            rate_txt = f'{r["rate"] * 100:.0f}% fraud ({r["n"]:,} similar)'
        body += f'<tr><td>{esc(r["check"])}</td><td>{esc(r["answer"])}</td><td>{esc(rate_txt)}</td><td>{pill}</td></tr>'
    return html_block(
        '<div class="card"><table class="tbl"><tr><th>Check</th><th>Your answer</th>'
        f'<th>Seen in dataset</th><th>Status</th></tr>{body}</table></div>'
    )


def confusion_matrix_html(cm: np.ndarray) -> str:
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return html_block(f"""
        <table class="cm">
          <tr><th></th><th>Predicted genuine</th><th>Predicted fraud</th></tr>
          <tr><th>Actual genuine</th><td class="good">{tn:,}</td><td class="badc">{fp:,}</td></tr>
          <tr><th>Actual fraud</th><td class="badc">{fn:,}</td><td class="good">{tp:,}</td></tr>
        </table>""")


def metric_row(m: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{m['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{m['recall'] * 100:.2f}%")
    c4.metric("F1-score", f"{m['f1'] * 100:.2f}%")


# ----------------------------------------------------------------------------------------
# 8. PAGES
# ----------------------------------------------------------------------------------------
def page_home() -> None:
    show_html("""
        <div class="hero">
          <h1>UPI Fraud Detection</h1>
          <div class="sub">Check the risk of a UPI transaction before you make it.</div>
          <div class="desc">Analyze transaction and receiver information using a machine-learning based
          fraud detection system.</div>
        </div>""")
    st.write("")
    st.button("Check a Transaction", type="primary", on_click=go_to, args=(PAGE_CHECK,))
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("🧾", "Transaction Analysis",
                     "Enter the amount, type and context of the payment you are about to make."), unsafe_allow_html=True)
    c2.markdown(card("🛡️", "Fraud Risk Detection",
                     "A Random Forest model trained on your dataset turns the details into a Fraud Risk Score."), unsafe_allow_html=True)
    c3.markdown(card("✅", "Safety Recommendation",
                     "Get a clear Low / Medium / High risk level with advice on what to do next."), unsafe_allow_html=True)
    st.write("")
    st.info(MODEL_DISCLAIMER)
    st.caption("🔒 " + PRIVACY_NOTE)


def validate_form(amount, receiver, upi_id, answers: dict, date_val, time_val) -> list[str]:
    errors = []
    if amount is None:
        errors.append("Please enter the transaction amount.")
    elif not np.isfinite(amount):
        errors.append("The amount is not a valid number.")
    elif amount < 0:
        errors.append("The amount cannot be negative.")
    elif amount == 0:
        errors.append("The amount must be greater than ₹0.")
    elif amount > MAX_ALLOWED_AMOUNT:
        errors.append(f"The amount is above the supported limit of {format_inr(MAX_ALLOWED_AMOUNT)}.")

    name = (receiver or "").strip()
    if not name:
        errors.append("Please enter the receiver / merchant name.")
    elif len(name) > 100:
        errors.append("The receiver name is too long (maximum 100 characters).")
    elif re.fullmatch(r"\d{4,8}", name):
        errors.append("The receiver name looks like a PIN/OTP. Never type a PIN or OTP here.")

    uid = (upi_id or "").strip()
    if uid and not re.fullmatch(r"[A-Za-z0-9._\-]{2,64}@[A-Za-z][A-Za-z0-9.\-]{1,63}", uid):
        errors.append("The UPI ID format looks wrong. It should look like name@bank.")

    if date_val is None or time_val is None:
        errors.append("Please choose the transaction date and time.")

    for label, value in answers.items():
        if value is None:
            errors.append(f"Please answer: {label}")
    return errors


def page_check(bundle: dict) -> None:
    st.title("Check a Transaction")

    if st.session_state.get("result"):
        render_result(bundle, st.session_state["result"])
        return

    show_html("""
        <div class="steps">
          <span class="step"><b>1</b> Transaction details</span>
          <span class="step"><b>2</b> Receiver details</span>
          <span class="step"><b>3</b> Payment context</span>
          <span class="step"><b>4</b> Analyze</span>
        </div>""")
    st.caption("🔒 " + PRIVACY_NOTE)
    st.caption("Nothing is paid or transferred here. This page only estimates fraud risk.")

    fid = st.session_state.setdefault("form_id", 0)
    opts = bundle["options"]
    stats = bundle["stats"]
    now = datetime.now()

    def labelled(feature: str) -> dict:
        return {value_label(feature, v): v for v in opts[feature]}

    type_map = labelled("transaction_type")
    cat_map = labelled("merchant_category_code")
    src_map = labelled("session_source")
    auth_map = labelled("authorization_method")
    yes_no = {"Yes": 1, "No": 0}
    tri_typo = {"Yes": "typo_squatting", "No": "none", "Not sure": None}
    tri_recent = {"Yes": "recent", "No": "none", "Not sure": None}

    with st.form(f"txn_form_{fid}"):
        # ---- Section A -------------------------------------------------------------------
        st.markdown('<div class="section-title">A · Transaction details</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            amount = c1.number_input("Transaction amount (₹)", value=None, step=100.0, format="%.2f",
                                     placeholder="e.g. 2500", key=f"amount_{fid}")
            type_label = c2.selectbox("Transaction type", list(type_map), index=None,
                                      placeholder="Select…", key=f"type_{fid}")
            c3, c4 = st.columns(2)
            date_val = c3.date_input("Date", value=now.date(), key=f"date_{fid}")
            time_val = c4.time_input("Time", value=now.replace(second=0, microsecond=0).time(), key=f"time_{fid}")
            st.caption("Only the hour of the day is used by the model - the dataset has no calendar-date information.")

        # ---- Section B -------------------------------------------------------------------
        st.markdown('<div class="section-title">B · Receiver details</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            receiver = c1.text_input("Receiver / merchant name", placeholder="e.g. Sharma Traders",
                                     max_chars=100, key=f"receiver_{fid}")
            upi_id = c2.text_input("Receiver UPI ID (optional)", placeholder="name@bank",
                                   max_chars=80, key=f"upi_{fid}")
            typo_label = st.radio("Does the UPI ID look like a misspelled copy of a known business or person?",
                                  list(tri_typo), index=None, horizontal=True, key=f"typo_{fid}")
            recent_label = st.radio("Was this UPI handle created recently?",
                                    list(tri_recent), index=None, horizontal=True, key=f"recent_{fid}")
            st.caption("Choose “Not sure” if you cannot tell - the score then uses the average over the "
                       "training data instead of a made-up value.")

        # ---- Section C -------------------------------------------------------------------
        st.markdown('<div class="section-title">C · Payment context</div>', unsafe_allow_html=True)
        with st.container(border=True):
            c1, c2 = st.columns(2)
            src_label = c1.selectbox("How did you reach this payment?", list(src_map), index=None,
                                     placeholder="Select…", key=f"src_{fid}")
            cat_label = c2.selectbox("Merchant category", list(cat_map), index=None,
                                     placeholder="Select…", key=f"cat_{fid}")
            auth_label = st.selectbox("How is this payment asked to be approved?", list(auth_map), index=None,
                                      placeholder="Select…", key=f"auth_{fid}")
            st.caption("Select the method only. Never type your UPI PIN or an OTP on this page.")
            urgent_label = st.radio("Does the request use urgent or pressuring words (e.g. “urgent”, “refund”, “act now”)?",
                                    list(yes_no), index=None, horizontal=True, key=f"urgent_{fid}")
            screen_label = st.radio("Has anyone asked you to install or use a screen-sharing / screen-mirroring app?",
                                    list(yes_no), index=None, horizontal=True, key=f"screen_{fid}")

        submitted = st.form_submit_button("Analyze Transaction", type="primary")

    if not submitted:
        return

    answers = {
        "Transaction type": type_label,
        "Does the UPI ID look like a misspelled copy?": typo_label,
        "Was the UPI handle created recently?": recent_label,
        "How did you reach this payment?": src_label,
        "Merchant category": cat_label,
        "Approval method": auth_label,
        "Urgent or pressuring words": urgent_label,
        "Screen-sharing app requested": screen_label,
    }
    errors = validate_form(amount, receiver, upi_id, answers, date_val, time_val)
    if errors:
        for message in errors:
            st.warning(message)
        return

    # Build the model input in the exact FEATURE_COLUMNS layout
    record = {
        "amount": float(amount),
        "hour_of_day": int(time_val.hour),
        "urgent_language": yes_no[urgent_label],
        "screen_sharing_requested": yes_no[screen_label],
        "transaction_type": type_map[type_label],
        "merchant_category_code": cat_map[cat_label],
        "session_source": src_map[src_label],
        "authorization_method": auth_map[auth_label],
        "handle_typo_analysis": tri_typo[typo_label],
        "handle_registration_pattern": tri_recent[recent_label],
    }
    try:
        score = predict_risk(bundle, record)
        indicators = build_indicators(bundle, record)
    except Exception:
        st.error("Sorry, the transaction could not be analyzed. Please check your inputs and try again.")
        return

    level = risk_level(score)
    st.session_state["result"] = {
        "score": score,
        "level": level,
        "indicators": indicators,
        "out_of_range": not (stats["amount_min"] <= amount <= stats["amount_max"]),
        "summary": {
            "Transaction amount": format_inr(amount),
            "Transaction type": value_label("transaction_type", record["transaction_type"]),
            "Receiver": receiver.strip(),
            "UPI ID": (upi_id or "").strip() or "-",
            "Date / time": datetime.combine(date_val, time_val).strftime("%d %b %Y, %I:%M %p"),
            "Risk level": RISK_LEVELS[level]["label"],
            "Fraud Risk Score": f"{score:.1f}%",
            "Model prediction": RISK_LEVELS[level]["prediction"],
        },
    }
    st.rerun()


def render_result(bundle: dict, res: dict) -> None:
    style = RISK_LEVELS[res["level"]]
    score = res["score"]
    fill = max(2.0, min(score, 100.0))

    show_html(f"""
        <div class="risk-hero" style="background:{style['bg']}">
          <div class="risk-label" style="color:{style['color']}">TRANSACTION ANALYSIS</div>
          <div style="font-size:1.6rem;font-weight:800;margin-top:.3rem">{style['icon']} {style['label']}</div>
          <div class="risk-score">{score:.0f}%<small>Fraud Risk</small></div>
          <div class="risk-pred">Prediction: <b>{style['prediction']}</b> &nbsp;·&nbsp; {style['headline']}</div>
          <div class="gauge">
            <div class="gauge-fill" style="width:{fill}%;background:{style['color']}"></div>
            <div class="gauge-tick" style="left:{LOW_RISK_MAX}%"></div>
            <div class="gauge-tick" style="left:{HIGH_RISK_MIN}%"></div>
            <span class="gauge-lbl" style="left:0%">0%</span>
            <span class="gauge-lbl" style="left:{LOW_RISK_MAX}%">{LOW_RISK_MAX:.0f}%</span>
            <span class="gauge-lbl" style="left:{HIGH_RISK_MIN}%">{HIGH_RISK_MIN:.0f}%</span>
            <span class="gauge-lbl" style="left:100%">100%</span>
          </div>
        </div>""")
    st.write("")

    show_html(f"""
        <div class="reco" style="border-color:{style['color']}">
          <div class="t" style="color:{style['color']}">RECOMMENDED ACTION</div>
          <div class="m">{esc(style['recommendation'])}</div>
        </div>""")

    if res["out_of_range"]:
        s = bundle["stats"]
        st.info(f"This amount is outside the range seen in the training data "
                f"({format_inr(s['amount_min'])} – {format_inr(s['amount_max'])}). "
                "The model treats it like the nearest amount it has seen.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="section-title">⚠ Risk indicators</div>', unsafe_allow_html=True)
        st.markdown(indicators_table(res["indicators"]), unsafe_allow_html=True)
        st.caption("“Seen in dataset” = how often that answer was fraudulent in your CSV. These are descriptive "
                   "statistics; the model weighs all answers together, so they do not explain this one score on their own.")
    with right:
        st.markdown('<div class="section-title">Transaction summary</div>', unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td style='color:#94a3b8'>{esc(k)}</td><td><b>{esc(v)}</b></td></tr>"
            for k, v in res["summary"].items()
        )
        st.markdown(html_block(f'<div class="card"><table class="tbl">{rows}</table></div>'), unsafe_allow_html=True)

    st.markdown('<div class="section-title">Important factors considered by the model</div>', unsafe_allow_html=True)
    st.markdown(importance_bars(bundle, top_n=8), unsafe_allow_html=True)
    st.caption("General Random Forest feature importance across the whole dataset - not an explanation of this "
               "individual prediction.")

    st.write("")
    st.button("Check Another Transaction", type="primary", on_click=reset_checker)
    st.caption(MODEL_DISCLAIMER)


def page_insights(bundle: dict) -> None:
    s, m = bundle["stats"], bundle["metrics"]
    st.title("Model Insights")

    st.markdown('<div class="section-title">Dataset (calculated from your fraud_dataset.csv)</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{s['rows']:,}")
    c2.metric("Columns / features", f"{s['columns']} / {s['features']}")
    c3.metric("Missing values", f"{s['missing_total']:,}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Genuine transactions", f"{s['genuine']:,}")
    c5.metric("Fraudulent transactions", f"{s['fraud']:,}")
    c6.metric("Fraud percentage", f"{s['fraud_pct']:.2f}%")
    if s["missing_cols"]:
        st.caption("Columns with missing values: " + ", ".join(f"{c} ({n:,})" for c, n in s["missing_cols"].items())
                   + " - none of these are model inputs.")

    st.markdown('<div class="section-title">Machine-learning model</div>', unsafe_allow_html=True)
    st.write("**Random Forest Classifier** - the model analyzes transaction-related patterns learned from historical "
             "UPI transactions and predicts whether a new transaction appears fraudulent. "
             f"It uses {N_TREES} trees, balanced class weights and `random_state={RANDOM_STATE}`; "
             f"{int((1 - TEST_SIZE) * 100)}% of the data was used for training and "
             f"{int(TEST_SIZE * 100)}% ({s['test_rows']:,} transactions) held back for testing.")

    st.markdown('<div class="section-title">Model performance (test set)</div>', unsafe_allow_html=True)
    metric_row(m)
    cm_col, note_col = st.columns([2, 3])
    with cm_col:
        st.markdown(confusion_matrix_html(m["cm"]), unsafe_allow_html=True)
    with note_col:
        st.info("**Recall matters most for fraud detection.** Recall is the share of real fraud the model catches - "
                "missing a fraudulent transaction can be costly. Precision shows how many alerts were real fraud.")

    if min(m["accuracy"], m["precision"], m["recall"], m["f1"]) >= 0.999:
        base = bundle["baseline_metrics"]
        leaky = bundle["leaky_columns"]
        st.warning(
            "**The test scores are (almost) perfect - this was investigated, not hidden.**\n\n"
            + (f"• Leakage check: {', '.join('`' + c + '`' for c in leaky)} reveals `is_fraud` on its own "
               "and is therefore **not** used by the model.\n\n" if leaky else "")
            + "• The model still scores ~100% without it, because the fraud patterns in this dataset follow clear, "
              "rule-like signatures (e.g. link-based payments, OTP requests, urgent refund wording, lookalike handles, "
              "unknown merchant category). A model using only amount and hour reaches "
              f"{base['accuracy'] * 100:.1f}% accuracy but only {base['recall'] * 100:.1f}% recall, "
              "so the pattern fields are doing the work.\n\n"
              "• Treat these numbers as a property of this dataset - real-world fraud is messier, and real-world "
              "performance would be lower."
        )

    st.markdown('<div class="section-title">Important factors considered by the model</div>', unsafe_allow_html=True)
    st.markdown(importance_bars(bundle, top_n=10), unsafe_allow_html=True)
    st.caption("General Random Forest feature importance (all one-hot columns summed back to the original field).")

    st.markdown('<div class="section-title">How the 65 columns are used</div>', unsafe_allow_html=True)
    for grp in bundle["column_groups"]:
        with st.expander(f"{grp['group']} - {len(grp['columns'])} column(s)"):
            st.write(grp["reason"])
            st.code(", ".join(grp["columns"]) if grp["columns"] else "(none)", language=None)

    with st.expander("Leakage scan - columns that (almost) reveal is_fraud by themselves"):
        scan = bundle["leak_scan"]
        top = scan[scan["purity"] >= 0.95]
        if top.empty:
            st.write("No low-cardinality column reaches 95% label purity.")
        for _, r in top.iterrows():
            used = r["column"] in REQUIRED_RAW_COLUMNS
            st.write(f"`{r['column']}` - {r['purity'] * 100:.2f}% label purity "
                     f"({'used as a model input' if used else 'not used'})")
        st.caption("Purity = how often 'guess the majority class of each value' matches is_fraud.")


def page_about(bundle: dict) -> None:
    st.title("About")
    st.markdown(
        """
**UPI Fraud Detection** is a college project that estimates the fraud risk of a UPI transaction *before* it is made.

**How it works**

1. You describe the payment: amount, type, time, receiver and a few context questions a customer can realistically answer.
2. The answers are converted into the exact feature layout the Random Forest was trained on (same preprocessing pipeline).
3. The model returns a probability of fraud, shown as the **Fraud Risk Score** (probability × 100).
4. The score is mapped to **Low / Medium / High** risk with a recommended action.

**Risk levels** (editable in the code): Low below {low:.0f}% · Medium {low:.0f}%–{high:.0f}% · High above {high:.0f}%.

**Good to know**

- The Fraud Risk Score is the model's suspicion level - **not** the probability that you will lose money.
- Fields like receiver name and UPI ID are shown in the summary and format-checked, but the dataset has no matching
  column, so they are not model inputs.
- The dataset's backend/device/behavioural columns (IP, typing speed, PIN entry, …) cannot be entered by a customer, so the
  deployable model does not use them.
- This app never processes payments and never asks for a UPI PIN, OTP, password or card details.
        """.format(low=LOW_RISK_MAX, high=HIGH_RISK_MIN)
    )
    st.info(MODEL_DISCLAIMER)
    st.warning(PRIVACY_NOTE)


# ----------------------------------------------------------------------------------------
# 9. MAIN
# ----------------------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="UPI Fraud Detection", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.session_state.setdefault("nav", PAGE_HOME)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("form_id", 0)

    with st.sidebar:
        st.markdown('<div class="side-title">🛡️ UPI Fraud Detection</div>'
                    '<div class="side-sub">Transaction safety checker</div>', unsafe_allow_html=True)
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.caption("**Dataset:** UPI Fraud Detection")
        st.caption("**Model:** Random Forest")

    # Load dataset (once) and train model (once)
    signature = CSV_PATH.stat().st_mtime if CSV_PATH.exists() else 0.0
    df, error = load_dataset(str(CSV_PATH), signature)
    if error:
        st.error(error)
        st.stop()
    try:
        bundle = train_bundle(df, signature)
    except Exception:
        st.error("The model could not be trained from fraud_dataset.csv. Please check that the file is the "
                 "original UPI Fraud Detection dataset.")
        st.stop()

    page = st.session_state["nav"]
    if page == PAGE_HOME:
        page_home()
    elif page == PAGE_CHECK:
        page_check(bundle)
    elif page == PAGE_INSIGHTS:
        page_insights(bundle)
    else:
        page_about(bundle)


if __name__ == "__main__":
    main()
