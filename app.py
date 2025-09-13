#!/usr/bin/env python3
"""
Run instructions
- Create a virtualenv (optional) and install dependencies:
    pip install streamlit pandas numpy plotly scipy python-dateutil pytz
- Run the app:
    streamlit run app.py

Notes
- This is a single-file Streamlit MVP for local use. Data is persisted to a JSON file in the project directory by default.
- A small synthetic dataset is embedded for quick demo on first run if no persistence exists and no CSV is provided.
- To use your own CSV, use the file uploader in the left panel or place a file at ./weights.csv.

CSV assumptions
- Columns: Date,Weight
- Date formats: ISO or common US; parsing is robust.
- Units: lbs; Timezone assumed local calendar days (America/Chicago semantics, but dates are treated naive).

"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
from dateutil import parser as dateparser
import html
import hashlib

# Supabase imports
from supabase import create_client, Client

# SciPy is optional for robust Theil–Sen. Fall back to simple OLS if unavailable.
try:
    from scipy.stats import theilslopes

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover
    SCIPY_AVAILABLE = False

import streamlit as st
import streamlit.components.v1 as components


# -------------------------------
# Configuration and constants
# -------------------------------
PERSIST_PATH = os.path.join(os.path.dirname(__file__), "weights.json")
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "weights.csv")
LOCAL_TZ = pytz.timezone("America/Chicago")

# Initialize Supabase client
try:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_ANON_KEY"]
    supabase: Client = create_client(url, key)
    SUPABASE_AVAILABLE = True
except Exception as e:
    supabase = None
    SUPABASE_AVAILABLE = False
    # Debug: Print the error for troubleshooting
    print(f"Supabase initialization failed: {e}")

# Handle OAuth callback at the top level
if SUPABASE_AVAILABLE and "code" in st.query_params:
    code = st.query_params["code"]
    try:
        res = supabase.auth.exchange_code_for_session({"auth_code": code})
        if res.user:
            st.session_state.user = res.user
            st.session_state.session = res.session
            st.success(f"✅ Logged in as {res.user.email}")
    except Exception as e:
        st.error(f"❌ Failed to exchange code: {e}")

# Persist session across reruns
if SUPABASE_AVAILABLE and "session" in st.session_state and st.session_state.session:
    try:
        supabase.auth.set_session(
            st.session_state.session.access_token,
            st.session_state.session.refresh_token
        )
    except Exception:
        pass  # Ignore session errors


def test_supabase_schema():
    """Test Supabase connection."""
    if not SUPABASE_AVAILABLE:
        return False
    
    try:
        # Try to get table info
        response = supabase.table("weight_logs").select("*").limit(1).execute()
        return True
    except Exception as e:
        return False

SAMPLE_CSV = """Date,Weight
2025-01-01,200.0
2025-01-03,199.2
2025-01-05,198.8
2025-01-08,198.6
2025-01-10,197.9
2025-01-12,197.2
2025-01-15,196.8
2025-01-18,196.0
2025-01-21,195.5
2025-01-25,195.1
"""


# -------------------------------
# Utility and core functions
# -------------------------------

def _ensure_date(obj) -> date:
    if isinstance(obj, pd.Timestamp):
        return obj.date()
    if isinstance(obj, datetime):
        return obj.date()
    if isinstance(obj, date):
        return obj
    # robust parse
    dt = dateparser.parse(str(obj)).date()
    return dt


def _clean_inplace(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean and normalize the dataset.
    - Parse Date as datetime64[ns], then keep only date part
    - Coerce Weight to float
    - Drop nulls
    - Deduplicate by Date keeping the last occurrence
    - Sort ascending by Date

    Returns: (clean_df, stats)
    stats = {"dropped_nulls": int, "deduped": int}
    """
    stats = {"dropped_nulls": 0, "deduped": 0}

    if "Date" not in df.columns or "Weight" not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Weight' columns")

    # Coerce types
    # Keep original length for stats
    orig_len = len(df)

    # Parse and coerce
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce").astype(float)

    # Drop rows with nulls after coercion
    df = df.dropna(subset=["Date", "Weight"]).reset_index(drop=True)
    stats["dropped_nulls"] = orig_len - len(df)

    # Aggregate multiple logs per calendar date by mean (calendar-day canonicalization)
    before = len(df)
    df = df.groupby("Date", as_index=False)["Weight"].mean()
    stats["deduped"] = before - len(df)

    # Sort ascending
    df = df.sort_values("Date").reset_index(drop=True)

    return df, stats


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset. See _clean_inplace for details.
    """
    cleaned, _ = _clean_inplace(df)
    return cleaned


def save_data(df: pd.DataFrame, path: str = PERSIST_PATH) -> None:
    # Persist to a simple JSON list for portability
    data = [{"Date": d.strftime("%Y-%m-%d"), "Weight": float(w)} for d, w in zip(df["Date"], df["Weight"])]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"weights": data, "saved_at": datetime.now().isoformat()}, f, indent=2)


def load_persisted(path: str = PERSIST_PATH) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        rows = obj.get("weights", [])
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["Date", "Weight"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        df["Weight"] = pd.to_numeric(df["Weight"]).astype(float)
        df, _ = _clean_inplace(df)
        return df
    except Exception:
        return None


def load_data(csv_path: Optional[str] = None, persisted_store: Optional[str] = PERSIST_PATH) -> pd.DataFrame:
    """
    Load dataset with priority: persisted store -> provided CSV -> default CSV path -> sample data.

    Returns a cleaned DataFrame with columns [Date (date), Weight (float)].
    """
    # Try persisted store
    if persisted_store:
        df = load_persisted(persisted_store)
        if df is not None and not df.empty:
            return df

    # Try provided CSV path
    candidate_paths: List[str] = []
    if csv_path:
        candidate_paths.append(csv_path)
    candidate_paths.append(DEFAULT_CSV_PATH)

    for p in candidate_paths:
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df, _ = _clean_inplace(df)
                return df
            except Exception:
                pass

    # Fallback: sample data
    df = pd.read_csv(StringIO(SAMPLE_CSV))
    df, _ = _clean_inplace(df)
    return df


def add_or_update_entry(df: pd.DataFrame, entry_date: date, weight_lbs: float) -> pd.DataFrame:
    """
    Add or update an entry for a specific date.
    """
    entry_date = _ensure_date(entry_date)
    weight_lbs = float(weight_lbs)

    df = df.copy()
    if (df["Date"] == entry_date).any():
        df.loc[df["Date"] == entry_date, "Weight"] = weight_lbs
    else:
        df = pd.concat([df, pd.DataFrame([{"Date": entry_date, "Weight": weight_lbs}])], ignore_index=True)
    df, _ = _clean_inplace(df)
    return df


def delete_entry(df: pd.DataFrame, entry_date: date) -> pd.DataFrame:
    entry_date = _ensure_date(entry_date)
    df = df.copy()
    df = df[df["Date"] != entry_date].reset_index(drop=True)
    df, _ = _clean_inplace(df)
    return df


def compute_rolling(df: pd.DataFrame, window: int = 7) -> pd.Series:
    """
    Compute a centered rolling average over logged days.

    >>> _df = pd.DataFrame({"Date": pd.date_range("2025-01-01", periods=7).date, "Weight": [1,2,3,4,5,6,7]})
    >>> compute_rolling(_df, 3).round(2).tolist()
    [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5]
    """
    if df.empty:
        return pd.Series(dtype=float)
    s = df["Weight"].rolling(window=window, center=True, min_periods=max(1, window // 2)).mean()
    # For edges where center cannot be applied, use trailing mean as a fallback
    s = s.combine_first(df["Weight"].rolling(window=window, center=False, min_periods=1).mean())
    return s


def _week_resample_rule(week_start: str) -> str:
    # Sunday–Saturday windows end Saturday => 'W-SAT'; Monday–Sunday windows end Sunday => 'W-SUN'
    if week_start.lower().startswith("sun"):
        return "W-SAT"
    return "W-SUN"


def _weekly_measure(series: pd.Series, count: pd.Series) -> Tuple[pd.Series, List[str]]:
    """
    Given weekly aggregated mean (series) and counts, choose weekly measure per definition:
    - If count >= 3, use mean
    - Else, use last value of that week (fallback) and record a note
    Returns: (measure_series, notes)
    """
    notes: List[str] = []
    measure = series.copy()
    # Identify weeks with low counts
    low_weeks = count[count < 3].index
    if len(low_weeks) > 0:
        notes.append(f"Weeks with <3 entries used last value fallback: {', '.join([d.strftime('%Y-%m-%d') for d in low_weeks])}")
    return measure, notes


def compute_weekly_metrics(df: pd.DataFrame, week_start: str = "Sunday") -> Dict[str, object]:
    """
    Compute weekly change metrics per definition.

    Returns dict with keys:
    - this_week_change
    - last_week_change
    - delta
    - notes (list of strings)
    - weekly_changes_df (for chart)

    Doctest: ensure produces a numeric change for simple synthetic data under both week starts
    >>> base = datetime(2025, 1, 1).date()
    >>> df = pd.DataFrame({
    ...     'Date': [base + timedelta(days=i) for i in range(14)],
    ...     'Weight': [200 - 0.5*i for i in range(14)]
    ... })
    >>> m1 = compute_weekly_metrics(df, 'Sunday')
    >>> m2 = compute_weekly_metrics(df, 'Monday')
    >>> isinstance(m1['this_week_change'], float) and isinstance(m2['this_week_change'], float)
    True
    """
    notes: List[str] = []
    if df.empty:
        return {"this_week_change": None, "last_week_change": None, "delta": None, "notes": ["No data"], "weekly_changes_df": pd.DataFrame()}

    # Work with a DateTimeIndex for resampling
    tmp = df.copy()
    tmp["Date"] = pd.to_datetime(tmp["Date"])  # midnight naive
    tmp = tmp.set_index("Date").sort_index()

    rule = _week_resample_rule(week_start)

    # Prepare weekly means, counts, and last values
    weekly_mean = tmp["Weight"].resample(rule).mean()
    weekly_count = tmp["Weight"].resample(rule).count()
    weekly_last = tmp["Weight"].resample(rule).last()

    # Fallback to last value if <3 entries
    measure = weekly_mean.copy()
    low_mask = weekly_count < 3
    if low_mask.any():
        measure[low_mask] = weekly_last[low_mask]
        low_weeks = weekly_count[low_mask].index
        notes.append(f"Weeks with <3 entries used last value fallback: {', '.join([d.strftime('%Y-%m-%d') for d in low_weeks])}")

    # Compute weekly change: current week measure minus prior week measure
    weekly_change = measure.diff()
    weekly_df = pd.DataFrame({
        "WeekEnd": weekly_change.index,
        "WeeklyChange": weekly_change.values,
        "WeeklyMeasure": measure.values,
        "Count": weekly_count.values,
    }).dropna(subset=["WeeklyChange"])  # drop first NaN diff

    if weekly_df.empty:
        # Not enough weeks to compare
        return {
            "this_week_change": None,
            "last_week_change": None,
            "delta": None,
            "notes": notes + ["Not enough weekly data to compute change"],
            "weekly_changes_df": pd.DataFrame(),
        }

    this_week_change = float(weekly_df.iloc[-1]["WeeklyChange"]) if len(weekly_df) >= 1 else None
    last_week_change = float(weekly_df.iloc[-2]["WeeklyChange"]) if len(weekly_df) >= 2 else None
    delta = (this_week_change - last_week_change) if (this_week_change is not None and last_week_change is not None) else None

    return {
        "this_week_change": this_week_change,
        "last_week_change": last_week_change,
        "delta": delta,
        "notes": notes,
        "weekly_changes_df": weekly_df,
    }


def _subset_by_days(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    if df.empty or not days:
        return df.copy()
    cutoff = df["Date"].max() - timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()


def compute_trend(df: pd.DataFrame, windows: List[Optional[int]] = [60, None]) -> Dict[Optional[int], Dict[str, float]]:
    """
    Compute robust linear trend slope in lbs/day and lbs/week for the given windows (in days) and overall (None).
    Returns a dict: {window: {slope_day, slope_week, r2}}

    Doctest: slope units
    >>> base = datetime(2025, 1, 1).date()
    >>> df = pd.DataFrame({'Date': [base + timedelta(days=i) for i in range(10)], 'Weight': [200 - 1*i for i in range(10)]})
    >>> t = compute_trend(df, windows=[None])[None]
    >>> round(t['slope_day'], 1) == -1.0 and round(t['slope_week'], 1) == -7.0
    True
    """
    results: Dict[Optional[int], Dict[str, float]] = {}

    def fit_trend(local_df: pd.DataFrame) -> Tuple[float, float, float]:
        if local_df is None or local_df.empty:
            return float("nan"), float("nan"), float("nan")
        x = pd.to_datetime(local_df["Date"]).map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        y = local_df["Weight"].to_numpy(dtype=float)
        if len(x) < 2:
            return float("nan"), float("nan"), float("nan")
        if SCIPY_AVAILABLE and len(x) >= 3:
            slope, intercept, _, _ = theilslopes(y, x)
        else:
            # Fallback to simple least squares
            slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        # R^2
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        slope_day = float(slope)
        slope_week = slope_day * 7.0
        return slope_day, slope_week, r2

    for w in windows:
        local_df = _subset_by_days(df, w) if w else df
        slope_day, slope_week, r2 = fit_trend(local_df)
        results[w] = {"slope_day": slope_day, "slope_week": slope_week, "r2": r2}

    return results


def compute_projection(df: pd.DataFrame, target_weight: Optional[float]) -> Dict[str, object]:
    """
    Using the last 28-day trend, estimate a projected date for reaching target_weight.
    Returns {projected_date|None, method, caveats}

    Doctest: projection logic for downward vs flat trend
    >>> base = datetime(2025, 1, 1).date()
    >>> df1 = pd.DataFrame({'Date': [base + timedelta(days=i) for i in range(28)], 'Weight': [200 - 0.5*i for i in range(28)]})
    >>> p1 = compute_projection(df1, 180.0)
    >>> p1['projected_date'] is not None
    True
    >>> df2 = pd.DataFrame({'Date': [base + timedelta(days=i) for i in range(28)], 'Weight': [200 for _ in range(28)]})
    >>> p2 = compute_projection(df2, 180.0)
    >>> p2['projected_date'] is None
    True
    """
    caveats: List[str] = []
    if target_weight is None or math.isnan(target_weight):
        return {"projected_date": None, "method": "none", "caveats": ["No target weight set"]}
    if df.empty:
        return {"projected_date": None, "method": "theil-sen" if SCIPY_AVAILABLE else "ols", "caveats": ["No data"]}

    last28 = _subset_by_days(df, 28)
    if last28.empty or len(last28) < 10:
        caveats.append("Low data volume in last 28 days; projection is unreliable")

    trends = compute_trend(df, windows=[28])
    slope_day = trends[28]["slope_day"]

    current_weight = float(df.iloc[-1]["Weight"]) if not df.empty else None

    if current_weight is None or np.isnan(slope_day):
        return {"projected_date": None, "method": "theil-sen" if SCIPY_AVAILABLE else "ols", "caveats": ["Insufficient data for trend"]}

    # If slope >= 0 (flat or gaining), no reliable projection
    if slope_day >= 0:
        return {"projected_date": None, "method": "linear-28d", "caveats": ["Trend is flat or gaining; no reliable projection"]}

    # If already at/below target, projection is today
    if current_weight <= target_weight:
        return {"projected_date": df.iloc[-1]["Date"], "method": "linear-28d", "caveats": caveats}

    days_needed = (target_weight - current_weight) / slope_day  # slope_day negative ⇒ positive days
    if days_needed < 0 or not np.isfinite(days_needed):
        return {"projected_date": None, "method": "linear-28d", "caveats": caveats + ["Unstable projection"]}

    projected = df.iloc[-1]["Date"] + timedelta(days=float(days_needed))
    return {"projected_date": projected, "method": "linear-28d", "caveats": caveats}


def compute_adherence(df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute logging consistency over last 30 days and longest streak over last 90 days.
    Returns {pct_last_30, streak_last_90}
    """
    if df.empty:
        return {"pct_last_30": 0.0, "streak_last_90": 0}

    today = df["Date"].max()

    # Last 30 days window inclusive
    start_30 = today - timedelta(days=29)
    mask_30 = (df["Date"] >= start_30) & (df["Date"] <= today)
    days_logged = df.loc[mask_30, "Date"].nunique()
    pct_30 = round(100.0 * days_logged / 30.0, 1)

    # Longest streak in last 90 days
    start_90 = today - timedelta(days=89)
    dates_90 = sorted(set(d for d in df["Date"] if start_90 <= d <= today))
    longest = 0
    current = 0
    prev = None
    for d in dates_90:
        if prev is None or (d - prev == timedelta(days=1)):
            current += 1
        else:
            longest = max(longest, current)
            current = 1
        prev = d
    longest = max(longest, current)

    return {"pct_last_30": pct_30, "streak_last_90": int(longest)}


def make_summary_text(metrics: Dict[str, object]) -> str:
    """
    Make a concise summary string from metrics bundle.

    This function expects a dict containing:
    - weekly: output from compute_weekly_metrics
    - trend_28: slope_week (last 28 days)
    - adherence: {pct_last_30, streak_last_90}
    - projection: {projected_date}

    >>> bundle = {
    ...   'weekly': {'this_week_change': -1.6, 'last_week_change': -0.8, 'delta': -0.8, 'notes': []},
    ...   'trend_28': {'slope_week': -1.1},
    ...   'adherence': {'pct_last_30': 90.0, 'streak_last_90': 10},
    ...   'projection': {'projected_date': date(2025, 10, 3)}
    ... }
    >>> s = make_summary_text(bundle)
    >>> "Past week:" in s and "28-day trend:" in s
    True
    """
    weekly = metrics.get("weekly", {})
    this_week = weekly.get("this_week_change")
    last_week = weekly.get("last_week_change")
    slope_week = metrics.get("trend_28", {}).get("slope_week")
    adherence = metrics.get("adherence", {})
    pct = adherence.get("pct_last_30")
    streak = adherence.get("streak_last_90")
    projection = metrics.get("projection", {})
    proj_date = projection.get("projected_date")

    parts = []
    if this_week is not None and last_week is not None:
        parts.append(f"Past week: {this_week:+.1f} lb (vs {last_week:+.1f} lb last week)")
    elif this_week is not None:
        parts.append(f"Past week: {this_week:+.1f} lb")
    if slope_week is not None and np.isfinite(slope_week):
        parts.append(f"28-day trend: {slope_week:+.1f} lb/week")
    if pct is not None and streak is not None:
        # Estimate days logged as round(pct/100*30)
        days_logged = int(round((pct / 100.0) * 30))
        parts.append(f"Logging consistency: {pct:.0f}% ({days_logged}/30 days)")
    if proj_date is not None:
        parts.append(f"Projected to target by {proj_date:%b %d, %Y}")
    else:
        if projection.get("caveats"):
            parts.append("No reliable projection")

    return ". ".join(parts) + "."


# Formatting helpers

def _format_change(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f} lb"


def _format_rate(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.2f} lb/wk"


def _format_std(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.2f} lb"


# Streamlit UI helpers

def _copy_button(text: str, label: str = "Copy summary"):
    """Render a one-click copy-to-clipboard button using a small HTML component."""
    escaped = (text or "").replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
    components.html(
        f"""
        <button id=\"copybtn\" style=\"padding:6px 10px; border:1px solid #ccc; border-radius:4px; background:#f6f6f6; cursor:pointer;\">{label}</button>
        <script>
        const btn = document.getElementById('copybtn');
        btn.addEventListener('click', async () => {{
          try {{
            await navigator.clipboard.writeText("{escaped}");
            btn.innerText = 'Copied!';
            setTimeout(()=>btn.innerText='{label}', 1500);
          }} catch(e) {{
            btn.innerText = 'Copy failed';
            setTimeout(()=>btn.innerText='{label}', 1500);
          }}
        }});
        </script>
        """,
        height=40,
    )


def _render_expandable_info(message: Optional[str], key_base: str) -> None:
    """
    Render a message inside an info-styled box that clamps long content with a gradient and a
    keyboard-accessible toggle preserving state across reruns.

    - Clamp threshold: >300 characters or >5 lines triggers the collapsed view
    - State key: f"{key_base}" (boolean)
    - Accessible toggle labels via button help text
    - Break long words to avoid overflow
    """
    if not message:
        return

    message_str = str(message)
    lines = message_str.splitlines() or [message_str]
    needs_clamp = (len(message_str) > 300) or (len(lines) > 5)

    if needs_clamp:
        first_lines = lines[:5]
        preview_concat = "\n".join(first_lines)
        if len(preview_concat) > 300:
            cut = preview_concat[:300]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            preview_concat = cut
        preview_text = preview_concat.rstrip()
    else:
        preview_text = message_str

    if key_base not in st.session_state:
        st.session_state[key_base] = False
    is_open = bool(st.session_state[key_base])

    box_style = (
        "border-left: 4px solid #1f77b4; "
        "background: rgba(31,119,180,0.08); "
        "padding: 10px 12px; border-radius: 4px;"
    )

    if not needs_clamp or is_open:
        safe = html.escape(message_str)
        st.markdown(
            f"<div style=\"{box_style}\"><div style='white-space:pre-wrap; word-break:break-word;'>{safe}</div></div>",
            unsafe_allow_html=True,
        )
        if needs_clamp:
            if st.button("Show less", key=f"{key_base}_less_btn", help="Collapse message", use_container_width=False):
                st.session_state[key_base] = False
                st.rerun()
    else:
        safe_prev = html.escape(preview_text)
        html_block = f"""
        <div style=\"{box_style} position: relative;\">
            <div style=\"white-space:pre-wrap; word-break:break-word; overflow:hidden; display:-webkit-box; -webkit-line-clamp:5; -webkit-box-orient:vertical;\">{safe_prev}</div>
            <div aria-hidden=\"true\" style=\"position:absolute; left:0; right:0; bottom:0; height:2.5em; 
                 background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0.92)); border-bottom-left-radius:4px; border-bottom-right-radius:4px;\"></div>
        </div>
        """
        st.markdown(html_block, unsafe_allow_html=True)
        if st.button("… Show more", key=f"{key_base}_more_btn", help="Expand message", use_container_width=False):
            st.session_state[key_base] = True
            st.rerun()


def _confidence_badges(df: pd.DataFrame, weekly_notes: List[str], trend_28_count: int):
    if len(df) < 7:
        st.warning("Fewer than 7 data points; rolling averages may be noisy.")
    if trend_28_count < 14:
        st.warning("Fewer than 14 logged days in the last 28; 28‑day metrics are low confidence.")
    for idx, note in enumerate(weekly_notes or []):
        key_base = f"fallback_weeks_message_open_{idx}_{hashlib.md5(note.encode('utf-8')).hexdigest()[:8]}"
        _render_expandable_info(note, key_base)


# -------------------------------
# Authentication and data persistence functions
# -------------------------------

def render_auth_ui():
    """Render authentication UI with login/signup/guest options."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "session" not in st.session_state:
        st.session_state.session = None
    
    
    if st.session_state.user is None:
        st.markdown("### Welcome to Gym Monster")
        st.markdown("**A comprehensive fitness tracking application with AI-powered insights**")
        
        # Google OAuth Login Button - Prominent placement
        if SUPABASE_AVAILABLE:
            st.markdown("---")
            st.markdown("### 🔑 Quick Login")
            
            # Determine redirect URL based on environment
            try:
                current_url = st.get_option("server.headless")
                if current_url or "localhost" in str(st.get_option("server.port")):
                    redirect_url = "http://localhost:8501"
                else:
                    redirect_url = "https://gym-monster.streamlit.app"
            except:
                redirect_url = "http://localhost:8501"
            
            auth_url = f"{st.secrets['SUPABASE_URL']}/auth/v1/authorize?provider=google&redirect_to={redirect_url}&flow_type=pkce"
            
            # Large, prominent Google login button
            google_col1, google_col2, google_col3 = st.columns([1, 2, 1])
            with google_col2:
                st.markdown(f"[🔑 **Login with Google**]({auth_url})", unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Prominent Demo Button for Recruiters
        st.markdown("### 🎯 For Recruiters & Demo")
        st.markdown("**Try the full application with sample data - no signup required!**")
        
        # Large, prominent demo button
        demo_col1, demo_col2, demo_col3 = st.columns([1, 2, 1])
        with demo_col2:
            if st.button("🚀 **DEMO THIS PROGRAM**", key="demo_btn", use_container_width=True, type="primary"):
                st.session_state.user = {"id": "guest", "email": "demo@example.com"}
                st.session_state.session = None  # No session for demo mode
                st.success("🎉 Welcome to the demo! You can now explore all features with sample data.")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### Other Authentication Options")
        st.markdown("Choose how you'd like to use the app:")
        
        tab1, tab2, tab3 = st.tabs(["Email Login", "Sign Up", "Continue as Guest"])
        
        with tab1:
            st.markdown("#### Login to Your Account")
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_btn"):
                if login_email and login_password:
                    try:
                        response = supabase.auth.sign_in_with_password({
                            "email": login_email,
                            "password": login_password
                        })
                        st.session_state.user = {
                            "id": response.user.id,
                            "email": response.user.email
                        }
                        st.session_state.session = response.session
                        st.success("Login successful!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Login failed: {str(e)}")
                else:
                    st.error("Please enter both email and password.")
        
        with tab2:
            st.markdown("#### Create New Account")
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_password")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            if st.button("Sign Up", key="signup_btn"):
                if signup_email and signup_password and signup_confirm:
                    if signup_password != signup_confirm:
                        st.error("Passwords don't match.")
                    else:
                        try:
                            response = supabase.auth.sign_up({
                                "email": signup_email,
                                "password": signup_password
                            })
                            st.session_state.user = {
                                "id": response.user.id,
                                "email": response.user.email
                            }
                            st.session_state.session = response.session
                            st.success("Account created successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Sign up failed: {str(e)}")
                else:
                    st.error("Please fill in all fields.")
        
        with tab3:
            st.markdown("#### Continue as Guest")
            st.info("Guest mode uses local storage. Your data will not be saved between sessions.")
            if st.button("Continue as Guest", key="guest_btn"):
                st.session_state.user = {"id": "guest", "email": "guest@example.com"}
                st.session_state.session = None  # No session for guest mode
                st.success("Welcome! You're now in guest mode.")
                st.rerun()
    
    else:
        # User is logged in - show welcome message and logout option
        user_email = st.session_state.user.get("email", "Unknown")
        
        # Create a header with user info and logout button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### Welcome, {user_email}")
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                if st.session_state.user.get("id") not in ["guest", "demo@example.com"]:
                    try:
                        supabase.auth.sign_out()
                    except Exception:
                        pass  # Ignore logout errors
                st.session_state.clear()
                st.success("Logged out successfully!")
                st.rerun()
        
        st.markdown("---")


def load_data_persistent(user_id: str) -> pd.DataFrame:
    """Load data based on user authentication status."""
    if user_id in ["guest", "demo@example.com"]:
        df = load_persisted()
        if df is None or df.empty:
            return pd.DataFrame(columns=["Date", "Weight"])
        return df
    else:
        try:
            response = supabase.table("weight_logs").select("*").eq("user_id", user_id).execute()
            df = pd.DataFrame(response.data)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                # Rename columns to match expected format
                df = df.rename(columns={"date": "Date", "weight": "Weight"})
                df["Date"] = df["Date"].dt.date
            else:
                df = pd.DataFrame(columns=["Date", "Weight"])
            return df
        except Exception as e:
            st.error(f"Failed to load data from database: {str(e)}")
            return pd.DataFrame(columns=["Date", "Weight"])


def upsert_entry_persistent(user_id: str, entry_date: date, weight_lbs: float) -> bool:
    """Add or update an entry based on user authentication status."""
    # Handle guest users (local storage only)
    if user_id in ["guest", "demo@example.com"]:
        # Use existing local logic
        df = load_persisted()
        if df is None:
            df = pd.DataFrame(columns=["Date", "Weight"])
        df = add_or_update_entry(df, entry_date, weight_lbs)
        save_data(df)
        return True
    else:
        # Handle authenticated users (Supabase) - using delete-then-insert method
        try:
            # Delete any existing entry for this user/date
            supabase.table("weight_logs").delete().eq("user_id", user_id).eq("date", str(entry_date)).execute()
            
            # Insert new entry
            response = supabase.table("weight_logs").insert({
                "user_id": user_id,
                "date": str(entry_date),
                "weight": float(weight_lbs)
            }).execute()
            
            return True
        except Exception as e:
            st.error(f"❌ Failed to save entry: {str(e)}")
            return False


def delete_entry_persistent(user_id: str, entry_date: date) -> bool:
    """Delete an entry based on user authentication status."""
    if user_id in ["guest", "demo@example.com"]:
        # Use existing local logic
        df = load_persisted()
        if df is None:
            df = pd.DataFrame(columns=["Date", "Weight"])
        df = delete_entry(df, entry_date)
        save_data(df)
        return True
    else:
        try:
            supabase.table("weight_logs").delete().eq("user_id", user_id).eq("date", str(entry_date)).execute()
            return True
        except Exception as e:
            st.error(f"Failed to delete entry: {str(e)}")
            return False


def import_csv_persistent(user_id: str, csv_data: str) -> bool:
    """Import CSV data based on user authentication status."""
    try:
        df = pd.read_csv(StringIO(csv_data))
        df, stats = _clean_inplace(df)
        
        if user_id in ["guest", "demo@example.com"]:
            # Use existing local logic
            save_data(df)
            return True
        else:
            # Bulk upsert to Supabase using unique constraint (user_id, date)
            records = [
                {"user_id": user_id, "date": str(row["Date"]), "weight": float(row["Weight"])}
                for _, row in df.iterrows()
            ]
            
            # Delete all existing records for this user, then insert all new ones
            supabase.table("weight_logs").delete().eq("user_id", user_id).execute()
            
            # Insert all records
            response = supabase.table("weight_logs").insert(records).execute()
            
            return True
    except Exception as e:
        st.error(f"Failed to import CSV: {str(e)}")
        return False


def export_csv_persistent(user_id: str) -> str:
    """Export CSV data based on user authentication status."""
    df = load_data_persistent(user_id)
    return df.to_csv(index=False) if not df.empty else "Date,Weight\n"


# -------------------------------
# Visualization helpers
# -------------------------------

def make_daily_chart(df: pd.DataFrame, show_roll7: bool = True) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="Daily Weight", template="plotly_white")
        return fig

    x = df["Date"]
    y = df["Weight"]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Daily", hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} lb"))

    if show_roll7:
        roll7 = compute_rolling(df, 7)
        fig.add_trace(go.Scatter(x=x, y=roll7, mode="lines", name="7-day avg"))

    fig.update_layout(
        title="Daily Weight & Rolling Averages",
        xaxis_title="Date",
        yaxis_title="Weight (lb)",
        hovermode="x unified",
        template="plotly_white",
        uirevision="daily_rolling_uirev",
    )
    return fig


def make_weekly_change_chart(weekly_changes_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if weekly_changes_df is None or weekly_changes_df.empty:
        fig.update_layout(title="Weekly Change", template="plotly_white")
        return fig

    x = weekly_changes_df["WeekEnd"]
    y = weekly_changes_df["WeeklyChange"]
    colors = ["#2ca02c" if v < 0 else "#d62728" for v in y]  # green loss, red gain
    fig.add_trace(go.Bar(x=x, y=y, marker_color=colors, name="Weekly change"))
    fig.update_layout(
        title="Weekly Change (vs prior week)",
        xaxis_title="Week End",
        yaxis_title="Δ Weight (lb)",
        template="plotly_white",
    )
    return fig


def get_roll7_series(df: pd.DataFrame) -> Dict[str, object]:
    """
    Calendar 7-day WEIGHTED average (no gaps):
    - Daily canonicalization: average multiple logs per day; reindex to continuous daily dates
    - For each day t, consider days d=0..6 (t-d). Raw weights w_raw[d] = 7-d. Mask missing days.
    - If at least one day exists in window, compute normalized weighted mean; else NaN.
    Returns {"series": roll7w (pd.Series indexed by date), "meta": {...}}
    """
    if df.empty:
        return {"series": pd.Series(dtype=float), "meta": {"calendar_window": 7, "weights": [7,6,5,4,3,2,1]}}
    daily = df.copy().dropna(subset=["Date", "Weight"]).groupby("Date", as_index=False)["Weight"].mean().sort_values("Date")
    idx = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    s = pd.Series(daily["Weight"].values, index=pd.to_datetime(daily["Date"]))
    s = s.reindex(idx)
    n = len(s)
    # Precompute raw weights for offsets 0..6
    w_raw = np.array([7, 6, 5, 4, 3, 2, 1], dtype=float)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        # window indices j from i-6..i
        j_start = max(0, i - 6)
        j_end = i
        window_vals = s.iloc[j_start:j_end+1].to_numpy(dtype=float)
        # corresponding offsets d where d = i - j
        d_vals = np.arange(j_end - j_start, -1, -1)  # from (i-j_start) down to 0
        # Map to weights: for each position j, offset d = i-j in [0..6]
        w_j = np.array([w_raw[int(i - (j_start + k))] for k in range(len(window_vals))], dtype=float)
        mask = ~np.isnan(window_vals)
        w_eff = w_j * mask.astype(float)
        w_sum = w_eff.sum()
        if w_sum > 0:
            out[i] = float(np.nansum((window_vals * w_eff) / w_sum))
        # else remain NaN
    roll7w = pd.Series(out, index=idx.date)
    return {"series": roll7w, "meta": {"calendar_window": 7, "weights": [7,6,5,4,3,2,1]}}


def _compute_date_range(df: pd.DataFrame, mode: str, custom_start: Optional[date], custom_end: Optional[date]) -> Tuple[Optional[date], Optional[date], Optional[str]]:
    """
    Return (start_date, end_date, error_message) based on preset logic.
    """
    if df.empty:
        return None, None, None
    min_d = df["Date"].min()
    max_d = df["Date"].max()

    if mode == "All":
        return min_d, max_d, None
    if mode == "YTD":
        start = date(max_d.year, 1, 1)
        return start, max_d, None

    days_map = {
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
    }
    if mode in days_map:
        start = max_d - timedelta(days=days_map[mode] - 1)
        return max(min_d, start), max_d, None

    if mode == "Custom":
        if custom_start is None or custom_end is None:
            return None, None, "Select start and end dates"
        if custom_start > custom_end:
            return None, None, "Start date must be on or before end date"
        # Clamp to dataset bounds
        return max(min_d, custom_start), min(max_d, custom_end), None

    # Fallback to All
    return min_d, max_d, None


def _autoscale_y(y_arrays: List[np.ndarray]) -> Tuple[float, float]:
    vals = np.concatenate([arr[~np.isnan(arr)] for arr in y_arrays if arr is not None and len(arr) > 0]) if y_arrays else np.array([])
    if vals.size == 0:
        return 0.0, 1.0
    y_min = float(np.min(vals))
    y_max = float(np.max(vals))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return 0.0, 1.0
    rng = y_max - y_min
    if rng <= 0.0:
        pad = 2.0
    else:
        pad = max(1.0, 0.05 * rng)
    y0 = max(50.0, y_min - pad)
    y1 = min(400.0, y_max + pad)
    if y1 - y0 < 2.0:
        # Ensure visible span
        mid = 0.5 * (y0 + y1)
        y0, y1 = mid - 1.0, mid + 1.0
    return y0, y1


def compute_avg_rate(df: pd.DataFrame, lookback_days: Optional[int] = None, start_date: Optional[date] = None, end_date: Optional[date] = None) -> Dict[str, object]:
    """
    Compute average change per day and per week using smoothed endpoints by default.
    - Window: [start_date, end_date] inclusive; if lookback_days provided, derive from max date
    - Smoothed endpoints: use roll-7 at the first/last in-range dates where roll-7 exists.
      If missing, try alt (min_periods=4). If still missing, fall back to raw weights at those dates.
    Returns: dict with status, avg_per_day, avg_per_week, start_date, end_date, days_elapsed,
             start_value_type (roll7|roll7_alt|raw), end_value_type, roll7_start, roll7_end,
             used_fallback_raw (bool), debug_lines
    """
    if df.empty:
        return {"status": "no_data"}
    df2 = df.copy().dropna(subset=["Date", "Weight"]).sort_values("Date")
    max_d = df2["Date"].max()
    if end_date is None:
        end_date = max_d
    if start_date is None:
        if lookback_days is None or lookback_days <= 0:
            start_date = df2["Date"].min()
        else:
            start_date = max(end_date - timedelta(days=int(lookback_days) - 1), df2["Date"].min())
    sub = df2[(df2["Date"] >= start_date) & (df2["Date"] <= end_date)].copy()
    sub = sub.sort_values("Date")
    debug_lines: List[str] = []
    lookback_label_days = (end_date - start_date).days + 1
    debug_lines.append(f"DEBUG AvgRate — Lookback:{lookback_label_days} start:{start_date} end:{end_date}")
    if len(sub) < 2:
        return {"status": "insufficient_data", "debug_lines": debug_lines}

    # Get roll-7 series computed on full dataset
    roll = get_roll7_series(df2)
    r_primary: pd.Series = roll["series"]  # indexed by date
    # Create alternative roll-7 with min_periods=4 for fallback
    daily = df2.copy().dropna(subset=["Date", "Weight"]).groupby("Date", as_index=False)["Weight"].mean().sort_values("Date")
    idx = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    s = pd.Series(daily["Weight"].values, index=pd.to_datetime(daily["Date"]))
    s = s.reindex(idx)
    r_alt = s.rolling(window=7, min_periods=4).mean()
    r_alt.index = r_alt.index.date
    # Build inclusive date list for window
    dates_window = pd.date_range(start_date, end_date, freq="D").date
    # Find first and last date in range with defined roll-7
    start_date_ep = next((d for d in dates_window if d in r_primary.index and pd.notna(r_primary.loc[d])), None)
    end_date_ep = next((d for d in reversed(dates_window.tolist()) if d in r_primary.index and pd.notna(r_primary.loc[d])), None)
    start_type = "roll7"
    end_type = "roll7"
    used_fallback_raw = False
    roll7_start = None
    roll7_end = None
    # Fallback to alt if needed
    if start_date_ep is None or end_date_ep is None:
        if start_date_ep is None:
            start_date_ep = next((d for d in dates_window if d in r_alt.index and pd.notna(r_alt.loc[d])), None)
            if start_date_ep is not None:
                start_type = "roll7_alt"
        if end_date_ep is None:
            end_date_ep = next((d for d in reversed(dates_window.tolist()) if d in r_alt.index and pd.notna(r_alt.loc[d])), None)
            if end_date_ep is not None:
                end_type = "roll7_alt"
    # If still missing, fall back to raw at endpoints
    if start_date_ep is None:
        start_date_ep = dates_window[0]
        start_type = "raw"
        used_fallback_raw = True
        roll7_start = float(df2.loc[df2["Date"] == start_date_ep, "Weight"].iloc[0]) if (df2["Date"] == start_date_ep).any() else None
    else:
        roll7_start = float(r_primary.loc[start_date_ep]) if start_type == "roll7" else float(r_alt.loc[start_date_ep])
    if end_date_ep is None:
        end_date_ep = dates_window[-1]
        end_type = "raw"
        used_fallback_raw = True or used_fallback_raw
        roll7_end = float(df2.loc[df2["Date"] == end_date_ep, "Weight"].iloc[0]) if (df2["Date"] == end_date_ep).any() else None
    else:
        roll7_end = float(r_primary.loc[end_date_ep]) if end_type == "roll7" else float(r_alt.loc[end_date_ep])

    endpoint_start_date = start_date_ep
    endpoint_end_date = end_date_ep
    if endpoint_start_date is None or endpoint_end_date is None:
        return {"status": "insufficient_data", "debug_lines": debug_lines}
    days_elapsed = (endpoint_end_date - endpoint_start_date).days
    debug_lines.append(f"w_start:{roll7_start} w_end:{roll7_end} days:{days_elapsed}")
    if used_fallback_raw:
        # count n_days present in the last 7 calendar days for visibility
        # build the 7-day window at endpoint where fallback happened
        if start_type == "raw":
            win = pd.date_range(endpoint_start_date - timedelta(days=6), endpoint_start_date, freq="D").date
            n_days = int(sum(1 for d in win if d in r_primary.index and pd.notna(r_primary.get(d, np.nan))))
            debug_lines.append(f"7D endpoint fallback at {endpoint_start_date}: used raw daily due to insufficient days (n={n_days}).")
        if end_type == "raw":
            win = pd.date_range(endpoint_end_date - timedelta(days=6), endpoint_end_date, freq="D").date
            n_days = int(sum(1 for d in win if d in r_primary.index and pd.notna(r_primary.get(d, np.nan))))
            debug_lines.append(f"7D endpoint fallback at {endpoint_end_date}: used raw daily due to insufficient days (n={n_days}).")
    if days_elapsed < 1:
        return {"status": "insufficient_data", "debug_lines": debug_lines}
    avg_day = (roll7_end - roll7_start) / days_elapsed
    avg_week = avg_day * 7.0
    debug_lines.append(f"avg/day:{avg_day} avg/week:{avg_week}")
    return {
        "status": "ok",
        "avg_per_day": float(round(avg_day, 6)),
        "avg_per_week": float(round(avg_week, 6)),
        "start_date": endpoint_start_date,
        "end_date": endpoint_end_date,
        "days_elapsed": int(days_elapsed),
        "w_start": roll7_start,
        "w_end": roll7_end,
        "start_value_type": start_type,
        "end_value_type": end_type,
        "used_fallback_raw": used_fallback_raw,
        "debug_lines": debug_lines,
    }


# -------------------------------
# Ask Gym Monster - Chat tools and panel
# -------------------------------

def tool_compute_metric(df: pd.DataFrame, metric_name: str, start_date: date, end_date: date, options: Optional[Dict] = None) -> Dict:
    options = options or {}
    # Implement a few core metrics
    data = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()
    data = data.sort_values("Date")
    if data.empty or len(data) <= 3:
        return {"ok": False, "error": "Insufficient data in range", "start": start_date, "end": end_date}
    if metric_name == "avg_daily_change":
        # Use start-to-end over elapsed days to avoid sampling bias
        res = compute_avg_rate(data, start_date=start_date, end_date=end_date)
        if res.get("status") != "ok":
            return {"ok": False, "error": "Cannot compute rate", "start": start_date, "end": end_date}
        return {"ok": True, "value_day": res["avg_per_day"], "value_week": res["avg_per_week"], "units": "lb/day", "start": res["start_date"], "end": res["end_date"], "method": "start-to-end over elapsed days"}
    if metric_name == "total_change":
        total = float(data["Weight"].iloc[-1] - data["Weight"].iloc[0])
        return {"ok": True, "value": total, "units": "lb", "start": start_date, "end": end_date, "method": "difference between first and last in range"}
    if metric_name == "biggest_week_drop":
        # Use current week_start setting Sunday for consistency here
        weekly = compute_weekly_metrics(df, week_start="Sunday").get("weekly_changes_df")
        if weekly is None or weekly.empty:
            return {"ok": False, "error": "No weekly data", "start": start_date, "end": end_date}
        w = weekly[(weekly["WeekEnd"].dt.date >= start_date) & (weekly["WeekEnd"].dt.date <= end_date)]
        if w.empty:
            return {"ok": False, "error": "No weekly data in range", "start": start_date, "end": end_date}
        idx = int(w["WeeklyChange"].idxmin())
        row = w.loc[idx]
        return {"ok": True, "week_end": row["WeekEnd"].date(), "change": float(row["WeeklyChange"]), "units": "lb", "start": start_date, "end": end_date, "method": "weekly change vs prior week"}
    return {"ok": False, "error": "Unknown metric", "start": start_date, "end": end_date}


def tool_describe_dataset(df: pd.DataFrame, snapshot_level: str = "brief") -> Dict:
    if df.empty:
        return {"ok": True, "count": 0}
    min_d = df["Date"].min()
    max_d = df["Date"].max()
    desc = {
        "ok": True,
        "count": int(len(df)),
        "date_min": min_d,
        "date_max": max_d,
        "min": float(df["Weight"].min()),
        "max": float(df["Weight"].max()),
        "mean": float(df["Weight"].mean()),
        "last_weigh_in": {"date": df.iloc[-1]["Date"], "weight": float(df.iloc[-1]["Weight"])},
    }
    if snapshot_level == "full":
        # Count missing calendar days in span
        all_days = pd.date_range(min_d, max_d, freq="D").date
        logged_days = set(df["Date"]) if not df.empty else set()
        desc["missing_days"] = int(sum(1 for d in all_days if d not in logged_days))
    return desc


def tool_project_goal_date(df: pd.DataFrame, current_weight: float, goal_weight: float, lookback_days: int, options: Optional[Dict] = None) -> Dict:
    options = options or {}
    if df.empty or lookback_days <= 0:
        return {"ok": False, "error": "No data"}
    # Use unified avg rate helper
    rate = compute_avg_rate(df, lookback_days=lookback_days)
    if rate.get("status") != "ok":
        return {"ok": False, "error": "Not enough recent data"}
    slope_day = rate["avg_per_day"]
    slope_week = rate["avg_per_week"]
    w_now = rate["w_end"]
    eps = 0.02
    if abs(slope_day) < eps:
        return {"ok": False, "error": "Insufficient trend to project", "slope_day": slope_day, "slope_week": slope_week}
    moving_towards = (w_now > goal_weight and slope_day < 0) or (w_now < goal_weight and slope_day > 0)
    if not moving_towards:
        return {"ok": False, "error": f"Not on track at current trend (gaining {slope_week:+.2f}/week)", "slope_day": slope_day, "slope_week": slope_week}
    days_to_goal = abs((w_now - goal_weight) / slope_day)
    if not np.isfinite(days_to_goal):
        return {"ok": False, "error": "Projection unstable", "slope_day": slope_day, "slope_week": slope_week}
    projected = rate["end_date"] + timedelta(days=int(math.ceil(days_to_goal)))
    years = (projected - rate["end_date"]).days / 365.25
    if years > 5:
        return {"ok": False, "error": "Projection too uncertain; widen lookback", "slope_day": slope_day, "slope_week": slope_week}
    return {"ok": True, "projected_date": projected, "slope_day": slope_day, "slope_week": slope_week, "lookback_days": lookback_days, "used_fallback_raw": bool(rate.get("used_fallback_raw"))}


def _render_chat_panel():
    st.markdown("Made with AI. Numbers come from your current dataset.")
    # Env guard
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        st.info("Still under devlopement, no OPENAI API key yet.")
        return

    st.session_state.setdefault("chat_history", [])  # list of {role, content}
    # Quick insertables
    inserts = [
        "What's my average change in the last 4 months?",
        "Project my goal date",
        "Which week had the biggest drop?",
        "Why was week X volatile?",
    ]
    ins_cols = st.columns(len(inserts))
    for i, txt in enumerate(inserts):
        with ins_cols[i]:
            if st.button(txt, key=f"ins_{i}"):
                st.session_state["chat_prefill"] = txt

    # Input
    prefill = st.session_state.get("chat_prefill", "")
    prompt = st.text_input("Ask Gym Monster", value=prefill, key="ask_input")
    ask = st.button("Ask", key="ask_send")

    # Minimal LLM wrapper (pseudo streaming via incremental write). We'll stub logic and use tools directly here.
    # A real implementation would call an LLM with a system prompt and tool schema, but here we route simple intents.
    if ask and prompt.strip():
        st.session_state["chat_prefill"] = ""
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        df = st.session_state.df
        min_d = df["Date"].min() if not df.empty else None
        max_d = df["Date"].max() if not df.empty else None

        # Simple intent routing for the required examples
        lower = prompt.lower()
        with st.chat_message("assistant"):
            if df.empty:
                st.write("I don't see any data yet. Upload a CSV to get started.")
            elif "average" in lower and ("4 months" in lower or "four months" in lower):
                start = max_d - timedelta(days=120)
                res = tool_compute_metric(df, "avg_daily_change", start, max_d)
                if not res.get("ok"):
                    st.write("Not enough data in that window. Try a longer range.")
                else:
                    st.write(f"Average rate from {start:%b %d, %Y} to {max_d:%b %d, %Y}: {res['value_day']:+.2f} lb/day ({res['value_week']:+.2f} lb/week).")
            elif "how much" in lower and ("july" in lower):
                # July of max year within dataset
                year = max_d.year
                start = date(year, 7, 1)
                end = date(year, 7, 31)
                res = tool_compute_metric(df, "total_change", start, end)
                if not res.get("ok"):
                    st.write("No sufficient data in July for that year.")
                else:
                    st.write(f"Change in July {year}: {res['value']:+.2f} lb (from {start:%b %d} to {end:%b %d}).")
            elif "biggest drop" in lower and "week" in lower:
                start = min_d
                end = max_d
                res = tool_compute_metric(df, "biggest_week_drop", start, end)
                if not res.get("ok"):
                    st.write("No weekly data available in that range.")
                else:
                    we = res["week_end"]
                    st.write(f"Biggest weekly drop ended {we:%b %d, %Y}: {res['change']:+.2f} lb.")
            elif "project" in lower and ("goal" in lower or "reach" in lower):
                current = float(df.iloc[-1]["Weight"]) if not df.empty else None
                goal = st.session_state.get("chat_goal_weight", 180.0)
                lookback = st.session_state.get("chat_lookback_days", 28)
                res = tool_project_goal_date(df, current, goal, lookback)
                if not res.get("ok"):
                    msg = res.get("error", "Cannot project with current data.")
                    st.write(msg)
                else:
                    d = res["projected_date"]
                    st.write(f"Projected to reach {goal:.1f} lb by {d:%b %d, %Y} based on last {res['lookback_days']} days ({res['slope_week']:+.2f} lb/week).")
                    if res.get("used_fallback_raw"):
                        st.info("Used raw endpoints due to insufficient roll‑7 at window edges.")
                    # DEBUG projection log
                    w_now = float(st.session_state.df.iloc[-1]["Weight"]) if not st.session_state.df.empty else None
                    days_to_goal = None
                    if res.get("slope_day") is not None and w_now is not None and res["slope_day"] != 0:
                        days_to_goal = abs((w_now - float(goal)) / res["slope_day"]) if np.isfinite(res["slope_day"]) else None
                    debug_proj = f"DEBUG Projection — w_now:{w_now} goal:{float(goal)} slope_day:{res.get('slope_day')} days_to_goal:{days_to_goal} projected_date:{d}"
                    st.code(debug_proj)
            else:
                st.write("I can answer questions about your rates, changes by month, biggest weekly drops, and goal projections. Try the insert buttons above.")


# Cache data between interactions
@st.cache_data(show_spinner=False)
def _load_initial_data(csv_path: Optional[str]) -> pd.DataFrame:
    return load_data(csv_path)


# Main UI
def main():
    st.set_page_config(page_title="Gym Monster", layout="wide")
    st.title("Gym Monster")
    st.caption("Your weight trends, explained.")
    
    # Check if Supabase is available
    if not SUPABASE_AVAILABLE:
        st.warning("⚠️ Supabase not configured. Running in guest mode only.")
        if "user" not in st.session_state:
            st.session_state.user = {"id": "guest", "email": "guest@example.com"}
    else:
        # Test Supabase connection
        test_supabase_schema()
    
    
    # Render authentication UI
    render_auth_ui()
    
    # If user is not authenticated, stop here
    if st.session_state.get("user") is None:
        return
    
    # State tracer for debugging (commented out - functionality confirmed)
    # st.session_state.setdefault("_run_id", 0)
    # st.session_state.setdefault("_last_commit", None)
    # st.session_state["_run_id"] += 1
    
    def _log(msg):
        pass  # Debug logging disabled
        # st.write(f"DBG r{st.session_state['_run_id']}: {msg}")
    
    # Transactional commit buffer (One Writer Rule)
    st.session_state.setdefault("_pending_commits", {})
    
    def commit_pending():
        pc = st.session_state["_pending_commits"]
        if pc:
            st.session_state.update(pc)   # atomic-ish
            st.session_state["_pending_commits"] = {}
            # _log(f"Applied pending commits: {list(pc.keys())}")
    
    # Initialize committed + UI state once
    if "committed_mode" not in st.session_state:
        st.session_state.update({
            "committed_mode": "6M",
            "committed_start": None,
            "committed_end": None,
            "ui_mode": "6M",
            "ui_start": None,
            "ui_end": None,
            "committed_weekly_mode": "6M",
            "committed_weekly_start": None,
            "committed_weekly_end": None,
            "ui_weekly_mode": "6M",
            "ui_weekly_start": None,
            "ui_weekly_end": None,
        })
        # _log("Initialized fresh state")
    
    # Helper to derive range from mode
    def derive_range_from_mode(df, mode):
        if df.empty:
            return None, None
        min_d = df["Date"].min()
        max_d = df["Date"].max()
        
        if mode == "All":
            return min_d, max_d
        if mode == "YTD":
            start = date(max_d.year, 1, 1)
            return start, max_d
        
        days_map = {"1W": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365}
        if mode in days_map:
            start = max_d - timedelta(days=days_map[mode] - 1)
            return max(min_d, start), max_d
        
        return None, None  # Custom mode
    
    # Callback functions (write only to _pending_commits)
    def on_preset_change():
        mode = st.session_state["ui_mode"]
        if mode != "Custom":
            # Load df to derive start/end
            df = st.session_state.get("df", pd.DataFrame())
            start, end = derive_range_from_mode(df, mode)
            st.session_state["_pending_commits"] = {
                "committed_mode": mode,
                "committed_start": start,
                "committed_end": end,
            }
            # _log(f"Preset change: {mode}")
    
    def on_apply_custom():
        start = st.session_state.get("ui_start")
        end = st.session_state.get("ui_end")
        if not start or not end or start > end:
            st.session_state["_pending_commits"] = {}  # no-op
            st.session_state["_last_commit"] = "invalid_custom"
            # _log("Invalid custom range")
            return
        st.session_state["_pending_commits"] = {
            "committed_mode": "Custom",
            "committed_start": start,
            "committed_end": end,
        }
        # _log(f"Custom range applied: {start} -> {end}")
    
    def on_weekly_preset_change():
        mode = st.session_state["ui_weekly_mode"]
        if mode != "Custom":
            df = st.session_state.get("df", pd.DataFrame())
            start, end = derive_range_from_mode(df, mode)
            st.session_state["_pending_commits"] = {
                "committed_weekly_mode": mode,
                "committed_weekly_start": start,
                "committed_weekly_end": end,
            }
            # _log(f"Weekly preset change: {mode}")
    
    def on_apply_weekly_custom():
        start = st.session_state.get("ui_weekly_start")
        end = st.session_state.get("ui_weekly_end")
        if not start or not end or start > end:
            st.session_state["_pending_commits"] = {}
            st.session_state["_last_commit"] = "invalid_weekly_custom"
            # _log("Invalid weekly custom range")
            return
        st.session_state["_pending_commits"] = {
            "committed_weekly_mode": "Custom",
            "committed_weekly_start": start,
            "committed_weekly_end": end,
        }
        # _log(f"Weekly custom range applied: {start} -> {end}")

    # Sidebar/left controls vs insights on right
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Data & Controls")

        # File import
        st.markdown("""Use the file uploader to import your CSV (Date,Weight). If omitted, the app will look for ./weights.csv or use a small sample.""")
        
        # Download link for example data
        if os.path.exists(os.path.join(os.path.dirname(__file__), "example_weight_data.csv")):
            with open(os.path.join(os.path.dirname(__file__), "example_weight_data.csv"), "rb") as f:
                csv_data = f.read()
            st.download_button(
                label="📥 Download larger example data set",
                data=csv_data,
                file_name="example_weight_data.csv",
                mime="text/csv",
                help="Download a larger example dataset to try out the app features"
            )
        
        uploaded = st.file_uploader("Import CSV", type=["csv"], accept_multiple_files=False)

        # Load initial data based on user authentication
        user_id = st.session_state.user["id"]
        
        if uploaded is not None:
            if "importing" not in st.session_state:
                st.session_state["importing"] = False

            if not st.session_state["importing"]:
                st.session_state["importing"] = True
                try:
                    # Read uploaded CSV
                    csv_content = uploaded.read().decode('utf-8')
                    if import_csv_persistent(user_id, csv_content):
                        st.success("✅ CSV imported successfully!")
                        # Reload data after import
                        df = load_data_persistent(user_id)
                    else:
                        st.error("❌ Failed to import CSV")
                        df = load_data_persistent(user_id)
                except Exception as e:
                    st.error(f"❌ Failed to read CSV: {e}")
                    df = load_data_persistent(user_id)
                finally:
                    st.session_state["importing"] = False
            else:
                # Still importing, show loading state
                st.info("🔄 Importing CSV...")
                df = load_data_persistent(user_id)
        else:
            # Load from persistent storage
            df = load_data_persistent(user_id)

        # Make df editable via our controls; persist changes
        if "df" not in st.session_state:
            st.session_state.df = df
        else:
            # If upload occurred, replace session df
            if uploaded is not None:
                st.session_state.df = df

        # Week toggle
        week_start = st.radio("Week start", ["Sunday", "Monday"], index=0, horizontal=True)

        # Target weight
        current_weight = float(st.session_state.df["Weight"].iloc[-1]) if not st.session_state.df.empty else None
        target_weight = st.number_input(
            "Target weight (lb)",
            value=float(round((current_weight - 10.0), 1)) if current_weight is not None else 180.0,
            step=0.1,
            format="%.1f",
        )
        # Lookback selection for projection
        st.session_state.setdefault("projection_lookback_mode", "28 days")
        st.session_state.setdefault("projection_custom_days", 28)
        lookback_options = ["21 days", "28 days", "60 days", "90 days", "Custom"]
        st.markdown("### Projection Lookback")
        st.session_state["projection_lookback_mode"] = st.selectbox(
            "Lookback window",
            lookback_options,
            index=lookback_options.index(st.session_state["projection_lookback_mode"]) if st.session_state["projection_lookback_mode"] in lookback_options else 1,
        )
        if st.session_state["projection_lookback_mode"] == "Custom":
            st.session_state["projection_custom_days"] = st.number_input("Custom lookback days", min_value=7, max_value=1000, value=st.session_state["projection_custom_days"], step=1)
        # Mirror to chat state
        lb_map = {"21 days": 21, "28 days": 28, "60 days": 60, "90 days": 90}
        st.session_state["chat_lookback_days"] = lb_map.get(st.session_state["projection_lookback_mode"], st.session_state.get("projection_custom_days", 28))
        st.session_state["chat_goal_weight"] = target_weight

        st.divider()
        st.markdown("### Add or Update Entry")
        entry_date = st.date_input("Date", value=date.today())
        entry_weight = st.number_input("Weight (lb)", value=float(current_weight) if current_weight is not None else 180.0, step=0.1, format="%.1f")
        
        # Check if entry exists for this date
        entry_exists = (st.session_state.df["Date"] == entry_date).any()
        if entry_exists:
            existing_weight = float(st.session_state.df.loc[st.session_state.df["Date"] == entry_date, "Weight"].iloc[0])
            st.info(f"📝 Entry exists for {entry_date}: {existing_weight:.1f} lb. Click 'Add / Update' to overwrite.")
        
        if st.button("Add / Update", use_container_width=True):
            if upsert_entry_persistent(user_id, entry_date, entry_weight):
                st.session_state.df = load_data_persistent(user_id)
                if entry_exists:
                    st.success(f"✅ Entry updated for {entry_date}: {entry_weight:.1f} lb")
                else:
                    st.success(f"✅ New entry added for {entry_date}: {entry_weight:.1f} lb")
            else:
                st.error("❌ Failed to save entry.")

        st.markdown("### Edit / Delete Existing")
        if not st.session_state.df.empty:
            dates = list(st.session_state.df["Date"].astype(str))
            selection = st.selectbox("Select date", options=dates, index=len(dates) - 1)
            if selection:
                sel_date = datetime.strptime(selection, "%Y-%m-%d").date()
                sel_weight = float(st.session_state.df.loc[st.session_state.df["Date"] == sel_date, "Weight"].iloc[0])
                new_weight = st.number_input("New weight (lb)", value=sel_weight, step=0.1, format="%.1f", key="edit_weight")
                cols = st.columns(2)
                with cols[0]:
                    if st.button("Save edit", use_container_width=True):
                        if upsert_entry_persistent(user_id, sel_date, new_weight):
                            st.session_state.df = load_data_persistent(user_id)
                            st.success(f"✅ Updated {sel_date}: {new_weight:.1f} lb")
                        else:
                            st.error("❌ Failed to save.")
                with cols[1]:
                    confirm_delete = st.checkbox("Confirm delete", value=False, key="confirm_delete")
                    if st.button("Delete", use_container_width=True):
                        if not confirm_delete:
                            st.warning("Please check 'Confirm delete' before deleting.")
                        else:
                            if delete_entry_persistent(user_id, sel_date):
                                st.session_state.df = load_data_persistent(user_id)
                                st.success(f"✅ Deleted entry for {sel_date}")
                            else:
                                st.error("❌ Failed to delete.")

        st.divider()
        # Export current dataset
        csv_data = export_csv_persistent(user_id)
        csv_bytes = csv_data.encode("utf-8")
        st.download_button("Export CSV", data=csv_bytes, file_name="weights_export.csv", mime="text/csv", use_container_width=True)

    with right:
        st.subheader("Insights")
        df = st.session_state.df


        # Weekly metrics
        weekly = compute_weekly_metrics(df, week_start=week_start)

        # Daily change volatility over last 28 logged days
        if len(df) >= 2:
            daily_change = df["Weight"].diff().dropna()
            last28_idx = max(0, len(daily_change) - 28)
            vol_28 = float(daily_change.iloc[last28_idx:].std()) if len(daily_change.iloc[last28_idx:]) > 0 else float("nan")
        else:
            vol_28 = float("nan")

        # Trends
        trends = compute_trend(df, windows=[60, None])
        trend_60 = trends.get(60, {"slope_day": float("nan"), "slope_week": float("nan"), "r2": float("nan")})
        trend_all = trends.get(None, {"slope_day": float("nan"), "slope_week": float("nan"), "r2": float("nan")})

        # Projection
        projection = compute_projection(df, target_weight)

        # Adherence
        adherence = compute_adherence(df)

        # Cards layout
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("This week change", _format_change(weekly.get("this_week_change")))
            st.metric("Last week change", _format_change(weekly.get("last_week_change")))
        with c2:
            st.metric("Delta (this - last)", _format_change(weekly.get("delta")))
            st.metric("Avg rate (28d)", _format_rate(compute_trend(df, windows=[28])[28]["slope_week"]))
        with c3:
            st.metric("Avg rate (overall)", _format_rate(trend_all.get("slope_week")))
            st.metric("Volatility (28d sd)", _format_std(vol_28))

        c4, c5 = st.columns(2)
        with c4:
            st.metric("Trend 60d", _format_rate(trend_60.get("slope_week")))
        with c5:
            st.metric("Trend overall", _format_rate(trend_all.get("slope_week")))

        # Projected Date to Goal card
        st.markdown("#### Projected Date to Goal")
        proj_col, info_col = st.columns([2,1])
        with proj_col:
            lookback_days = st.session_state.get("chat_lookback_days", 28)
            if target_weight is None or not np.isfinite(target_weight):
                st.write("Set a target to see a projection.")
            elif st.session_state.df.empty:
                st.write("No data.")
            else:
                current = float(st.session_state.df.iloc[-1]["Weight"]) if not st.session_state.df.empty else None
                # Already at/beyond goal within tolerance
                tol = 0.5
                if current is not None and current <= target_weight + tol:
                    reached_date = st.session_state.df.loc[st.session_state.df["Weight"] <= target_weight + tol, "Date"].max()
                    st.success(f"Goal reached 🎉 on {reached_date:%b %d, %Y}")
                else:
                    res = tool_project_goal_date(st.session_state.df, current, float(target_weight), int(lookback_days))
                    if not res.get("ok"):
                        msg = res.get("error", "Not enough recent data to project.")
                        st.info(msg)
                    else:
                        d = res["projected_date"]
                        slope_week = res["slope_week"]
                        st.metric("Projected Date", f"{d:%b %d, %Y}")
                        st.caption(f"Based on last {res['lookback_days']} days ({slope_week:+.2f} lb/week).")
                        if res.get("used_fallback_raw"):
                            st.info("Used raw endpoints due to insufficient roll‑7 at window edges.")
                        # DEBUG projection log
                        w_now = float(st.session_state.df.iloc[-1]["Weight"]) if not st.session_state.df.empty else None
                        days_to_goal = None
                        if res.get("slope_day") is not None and w_now is not None and res["slope_day"] != 0:
                            days_to_goal = abs((w_now - float(target_weight)) / res["slope_day"]) if np.isfinite(res["slope_day"]) else None
                        debug_proj = f"DEBUG Projection — w_now:{w_now} goal:{float(target_weight)} slope_day:{res.get('slope_day')} days_to_goal:{days_to_goal} projected_date:{d}"
                        st.code(debug_proj)
        with info_col:
            st.info("Linear projection based on your recent average change. Informational only.")

        # Average rate (selected lookback) debug block
        st.markdown("#### Average Rate (Selected Lookback)")
        lb = st.session_state.get("chat_lookback_days", 28)
        rate = compute_avg_rate(st.session_state.df, lookback_days=int(lb))
        if rate.get("status") == "no_data":
            st.info("No data.")
        elif rate.get("status") == "insufficient_data":
            st.info("Not enough data to compute rate.")
            dbg = "\n".join(rate.get("debug_lines", []))
            st.code(dbg)
        else:
            st.metric("Avg change/week", f"{rate['avg_per_week']:+.2f} lb/week")
            st.caption(f"Based on last {lb} days ({rate['start_date']}–{rate['end_date']}).")
            dbg = "\n".join(rate.get("debug_lines", []))
            st.code(dbg)

        # Confidence badges and notes
        trend_28_count = len(_subset_by_days(df, 28))
        _confidence_badges(df, weekly.get("notes", []), trend_28_count)

        st.divider()

        # Unified summary using active lookback and smoothed endpoints
        lb = st.session_state.get("chat_lookback_days", 28)
        rate_sum = compute_avg_rate(df, lookback_days=int(lb))
        proj_sum = tool_project_goal_date(df, float(df.iloc[-1]["Weight"]) if not df.empty else float("nan"), target_weight, int(lb))
        # Logging consistency over active window
        if rate_sum.get("status") == "ok":
            s_d = rate_sum["start_date"]
            e_d = rate_sum["end_date"]
            span_days = (e_d - s_d).days + 1
            dates_in = pd.date_range(s_d, e_d, freq="D").date
            num_logged = int(sum(1 for d0 in dates_in if d0 in set(df["Date"])))
            pct_log = round(100.0 * num_logged / span_days, 0)
        else:
            s_d = e_d = None
            pct_log = 0.0
            num_logged = 0
            span_days = 0
        trend_week = (rate_sum.get("avg_per_week") if rate_sum.get("status") == "ok" else None)
        proj_date = (proj_sum.get("projected_date") if proj_sum.get("ok") else None)
        # Weekly comparisons using roll-7 deltas
        r = get_roll7_series(df)
        rser = r["series"]
        roll_end = rate_sum.get("end_date") if rate_sum.get("status") == "ok" else (df["Date"].max() if not df.empty else None)
        notes = []
        this_week_change = last_week_change = None
        if roll_end is not None:
            # find indices by date
            idx_map = {d: i for i, d in enumerate(df["Date"]) }
            if roll_end in idx_map:
                i_end = idx_map[roll_end]
                def r_at(delta_days: int) -> Optional[float]:
                    # get roll-7 at exact date - delta_days, else None
                    target = roll_end - timedelta(days=delta_days)
                    return float(rser.iloc[idx_map[target]]) if target in idx_map and pd.notna(rser.iloc[idx_map[target]]) else None
                r0 = r_at(0)
                r7 = r_at(7)
                r14 = r_at(14)
                if r0 is not None and r7 is not None:
                    this_week_change = r0 - r7
                if r7 is not None and r14 is not None:
                    last_week_change = r7 - r14
                if this_week_change is None or last_week_change is None:
                    notes.append("Rolling average endpoints missing; weekly comparison skipped.")
        summary_payload = {
            "weekly": {"this_week_change": this_week_change, "last_week_change": last_week_change, "delta": (this_week_change - last_week_change) if this_week_change is not None and last_week_change is not None else None, "notes": notes},
            "trend_lookback": {"slope_week": trend_week},
            "adherence": {"pct_last_30": pct_log, "streak_last_90": None},
            "projection": {"projected_date": proj_date},
        }
        summary_text = make_summary_text(summary_payload)
        st.text_area("Summary", value=summary_text, height=60)
        ask_clicked = st.button("Ask Gym Monster", help="Open chat and prefill with this summary", use_container_width=False)
        if ask_clicked:
            st.session_state.setdefault("ask_gm_open", True)
            st.session_state["ask_gm_open"] = True
            st.session_state.setdefault("chat_prefill", "")
            st.session_state["chat_prefill"] = summary_text

        # Chat panel (Ask Gym Monster)
        with st.expander("Ask Gym Monster", expanded=st.session_state.get("ask_gm_open", False)):
            _render_chat_panel()

    st.divider()

    # Main area: charts and table
    st.subheader("Charts & Table")

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Range")
        modes = ["1W", "1M", "3M", "6M", "1Y", "YTD", "All", "Custom"]
        
        # Render preset selector (UI state only, callback handles commits)
        st.radio(
            "Date range",
            options=modes,
            key="ui_mode",
            on_change=on_preset_change,
            horizontal=True,
        )
        
        # Custom date controls
        if st.session_state.ui_mode == "Custom":
            cstart, cend = st.columns(2)
            with cstart:
                st.date_input("Start", key="ui_start")
            with cend:
                st.date_input("End", key="ui_end")
            
            # Apply button for custom range
            st.button("Apply Custom Range", on_click=on_apply_custom, key="apply_custom_btn")
            
            # Show validation errors
            if st.session_state.get("_last_commit") == "invalid_custom":
                st.error("Please select valid dates where start ≤ end")

        # Commit pending changes before computing data (ONE WRITER RULE)
        commit_pending()
        
        # _log(f"committed={st.session_state['committed_mode']} "
        #      f"{st.session_state.get('committed_start')}→{st.session_state.get('committed_end')}")

        # Compute range using only committed values (pure function)
        start_d, end_d, err = _compute_date_range(st.session_state.df, st.session_state.committed_mode, st.session_state.committed_start, st.session_state.committed_end)
        if err:
            st.error(err)
        else:
            # Prepare series: compute rolling on full data then filter
            df_full = st.session_state.df
            # Calendar 7-day series
            roll7_obj = get_roll7_series(df_full)
            roll7_full = roll7_obj["series"]

            mask = (df_full["Date"] >= start_d) & (df_full["Date"] <= end_d)
            df_vis = df_full.loc[mask].reset_index(drop=True)

            if df_vis.empty:
                st.warning("No data in selected range")
            else:
                # Visible arrays for autoscale
                y_arrays = [df_vis["Weight"].to_numpy(dtype=float)]
                if roll7_full is not None:
                    y_arrays.append(pd.Series(roll7_full).loc[start_d:end_d].to_numpy(dtype=float))
                # Y-axis mode: Auto | Manual
                st.session_state.setdefault("daily_y_mode", "Auto")
                st.session_state.setdefault("daily_y_min", 100.0)
                st.session_state.setdefault("daily_y_max", 200.0)
                st.session_state.daily_y_mode = st.radio("Y-axis mode", ["Auto", "Manual"], index=0 if st.session_state.daily_y_mode == "Auto" else 1, horizontal=True, key="daily_y_mode_radio")
                if st.session_state.daily_y_mode == "Manual":
                    cmin, cmax = st.columns(2)
                    with cmin:
                        st.session_state.daily_y_min = st.number_input("Y min", value=float(st.session_state.daily_y_min))
                    with cmax:
                        st.session_state.daily_y_max = st.number_input("Y max", value=float(st.session_state.daily_y_max))
                    y0, y1 = float(st.session_state.daily_y_min), float(st.session_state.daily_y_max)
                    if st.button("Reset to Auto (daily)"):
                        st.session_state.daily_y_mode = "Auto"
                        st.rerun()
                else:
                    y0, y1 = _autoscale_y(y_arrays)


                # Build figure
                fig = go.Figure()
                x = df_vis["Date"]
                y = df_vis["Weight"]
                
                # Daily trace (always first, stable identity)
                fig.add_trace(go.Scatter(
                    x=x, 
                    y=y, 
                    mode="lines+markers", 
                    name="Daily", 
                    uid="trace-daily",
                    legendgroup="daily",
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} lb"
                ))

                # Rolling 7 (always second, stable identity)
                roll7_vis = pd.Series(roll7_full).loc[start_d:end_d]
                x7 = list(roll7_vis.index)
                fig.add_trace(go.Scatter(
                    x=x7, 
                    y=roll7_vis.values, 
                    mode="lines", 
                    name="7-day avg", 
                    uid="trace-roll7",
                    legendgroup="roll7",
                    line=dict(width=3),
                    connectgaps=False
                ))
                
                # Set initial visibility (both visible by default)
                fig.data[0].visible = True  # Daily
                fig.data[1].visible = True  # 7-day avg
                # 7D gap connectors
                v7 = roll7_vis.values
                t7 = np.array(x7)
                if len(v7) > 0:
                    nan_mask = np.isnan(v7)
                    if nan_mask.any():
                        # find contiguous NaN runs and bounded neighbors
                        in_gap = False
                        start_nan = None
                        for k in range(len(v7)):
                            if nan_mask[k] and not in_gap:
                                in_gap = True
                                start_nan = k
                            if (not nan_mask[k] or k == len(v7)-1) and in_gap:
                                end_nan = k-1 if not nan_mask[k] else k
                                # bounded if start_nan>0 and end_nan < len(v7)-1
                                if start_nan > 0 and end_nan < len(v7)-1:
                                    i_end = start_nan - 1
                                    i_start = end_nan + 1
                                    fig.add_trace(go.Scatter(
                                        x=[t7[i_end], t7[i_start]],
                                        y=[v7[i_end], v7[i_start]],
                                        mode="lines",
                                        line=dict(dash="dot", width=3, color="red"),
                                        opacity=0.7,
                                        name="7-day connectors",
                                        uid="trace-roll7-connectors",
                                        legendgroup="roll7",
                                        showlegend=False,
                                        hoverinfo="skip",
                                    ))
                                in_gap = False
                                start_nan = None

                fig.update_layout(
                    title="Daily Weight & Rolling Averages",
                    xaxis_title="Date",
                    yaxis_title="Weight (lb)",
                    hovermode="x unified",
                    template="plotly_white",
                    yaxis=dict(range=[y0, y1]),
                    uirevision="daily_rolling_uirev",
                    legend=dict(groupclick="togglegroup"),
                )

                # Inline range summary with reset hint (Plotly has a reset in modebar)
                st.caption(f"Showing: {start_d:%b %d, %Y} – {end_d:%b %d, %Y}")
                st.plotly_chart(fig, key="daily_rolling_chart", use_container_width=True)
    with chart_col2:
        st.markdown("#### Weekly Range")
        modes_w = ["1W", "1M", "3M", "6M", "1Y", "YTD", "All", "Custom"]
        
        # Render weekly preset selector (UI state only, callback handles commits)
        st.radio(
            "Weekly date range",
            options=modes_w,
            key="ui_weekly_mode",
            on_change=on_weekly_preset_change,
            horizontal=True,
        )

        # Custom date controls
        if st.session_state.ui_weekly_mode == "Custom":
            csa, cea = st.columns(2)
            with csa:
                st.date_input("Start (weekly)", key="ui_weekly_start")
            with cea:
                st.date_input("End (weekly)", key="ui_weekly_end")
            
            # Apply button for custom weekly range
            st.button("Apply Custom Weekly Range", on_click=on_apply_weekly_custom, key="apply_weekly_custom_btn")
            
            # Show validation errors
            if st.session_state.get("_last_commit") == "invalid_weekly_custom":
                st.error("Please select valid dates where start ≤ end")

        # Commit weekly pending changes before computing data
        commit_pending()

        # Compute range using only committed values (pure function)
        w_start_d, w_end_d, w_err = _compute_date_range(st.session_state.df, st.session_state.committed_weekly_mode, st.session_state.committed_weekly_start, st.session_state.committed_weekly_end)
        if w_err:
            st.error(w_err)
        else:
            weekly = compute_weekly_metrics(st.session_state.df, week_start=week_start)
            wdf = weekly.get("weekly_changes_df")
            if wdf is None or wdf.empty:
                st.warning("No weekly data.")
            else:
                # Filter to selected weekly range by WeekEnd date
                wmask = (wdf["WeekEnd"].dt.date >= w_start_d) & (wdf["WeekEnd"].dt.date <= w_end_d)
                wdf_vis = wdf.loc[wmask]
                if wdf_vis.empty:
                    st.warning("No data in selected weekly range")
                else:
                    # Y-axis mode: Auto or Manual
                    st.session_state.setdefault("weekly_y_mode", "Auto")
                    st.session_state.setdefault("weekly_y_min", -5.0)
                    st.session_state.setdefault("weekly_y_max", 5.0)
                    st.session_state.weekly_y_mode = st.radio("Y-axis mode", ["Auto", "Manual"], index=0 if st.session_state.weekly_y_mode == "Auto" else 1, horizontal=True, key="weekly_y_mode_radio")
                    y0, y1 = None, None
                    if st.session_state.weekly_y_mode == "Manual":
                        cmin, cmax = st.columns(2)
                        with cmin:
                            st.session_state.weekly_y_min = st.number_input("Y min (weekly)", value=float(st.session_state.weekly_y_min))
                        with cmax:
                            st.session_state.weekly_y_max = st.number_input("Y max (weekly)", value=float(st.session_state.weekly_y_max))
                        y0, y1 = float(st.session_state.weekly_y_min), float(st.session_state.weekly_y_max)
                        if st.button("Reset to Auto (weekly)"):
                            st.session_state.weekly_y_mode = "Auto"
                            st.rerun()
                    else:
                        # Autoscale from visible bars
                        y_vals = wdf_vis["WeeklyChange"].to_numpy(dtype=float)
                        ymin = float(np.nanmin(y_vals))
                        ymax = float(np.nanmax(y_vals))
                        rng = ymax - ymin
                        pad = max(1.0, 0.05 * rng)
                        y0, y1 = ymin - pad, ymax + pad

                    fig_w = make_weekly_change_chart(wdf_vis)
                    fig_w.update_layout(yaxis=dict(range=[y0, y1]))
                    st.plotly_chart(fig_w, use_container_width=True)
 
    # Table view (read-only; edits done via controls above for clarity/persistence)
    st.dataframe(st.session_state.df, use_container_width=True, hide_index=True)


# -------------------------------
# Lightweight tests (doctests)
# -------------------------------

def _run_doctests_if_requested():
    import os as _os
    if _os.environ.get("RUN_DOCTESTS", "0") == "1":
        import doctest as _doctest
        _doctest.testmod(verbose=True)


if __name__ == "__main__":
    _run_doctests_if_requested()
    # Streamlit apps are started via: streamlit run app.py
    # Running this file directly won't start the UI, but doctests may run.
    pass

# Call the Streamlit app entrypoint when the script is executed by Streamlit
main() 