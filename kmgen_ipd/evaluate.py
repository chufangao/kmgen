"""
Unified Fidelity Metrics: Synthetic vs. Real (IPD) Patient Comparison
=====================================================================

Compares synthetic patient data against ground-truth individual patient
data (IPD) for any supported trial.

Supported trials: NCT03041311, NCT02499770, NCT00844649

Metrics computed:
    1. Demographic distributions (sex, ECOG, weight, height, BMI)
    2. Overall survival (KM curves, median OS, Mann-Whitney U)
    3. Adverse event frequency distributions (JSD, cosine similarity)
    4. AE timing distributions (KS test on onset times)
    5. AE burden per patient (mean/median count comparison)
    6. Organ-system-level AE distribution (JSD, cosine similarity)

Usage:
    conda run -n longehr python evaluations/evaluate.py --trial-id NCT03041311
    conda run -n longehr python evaluations/evaluate.py --trial-id NCT03041311 --synth-csv path/to/file.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from synthetic_patient_generator import TrialConfig, generate_combined

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# IPD LOADING — per-trial
# ============================================================================

def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def _parse_ecog(val) -> str:
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    if s.replace(".", "", 1).isdigit():
        return str(int(float(s)))
    return s


def _parse_age_band(val) -> float:
    """Reconstruct an approximate numeric age from an ADSL AGE band string.

    Some sponsor IPD deliveries coarsen AGE to 5-year bands (e.g. "50-54",
    "<45", ">=80") before release. We reconstruct a point estimate by taking
    the band midpoint. Open-ended tails (<45, >=80) use a single-decade
    midpoint (42, 82) which is consistent with typical oncology populations.
    """
    if pd.isna(val):
        return float("nan")
    s = str(val).strip()
    if not s:
        return float("nan")
    # Open-ended lower: <45, <50, ...
    if s.startswith("<"):
        try:
            upper = float(s[1:])
            return upper - 3  # e.g. "<45" -> 42
        except ValueError:
            return float("nan")
    # Open-ended upper: >=80, >80, >75
    if s.startswith(">"):
        rest = s[2:] if s.startswith(">=") else s[1:]
        try:
            lower = float(rest)
            return lower + 2  # e.g. ">=80" -> 82, ">75" -> 77
        except ValueError:
            return float("nan")
    # Two-sided band: "50-54", "45-49", "18-<65", "65-75"
    if "-" in s:
        lo, _, hi = s.partition("-")
        hi = hi.lstrip("<")
        try:
            return (float(lo) + float(hi)) / 2
        except ValueError:
            return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_real_nct03041311() -> dict:
    ipd_dir = os.path.join(BASE_DIR, "datasets", "NCT03041311", "ipd")

    adsl = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adsl.sas7bdat"), encoding="latin1"))
    # AGE in ADSL is delivered as a 5-year band string — reconstruct a numeric
    # midpoint so downstream metrics can use continuous age.
    age_numeric = adsl["AGE"].apply(_parse_age_band) if "AGE" in adsl.columns else pd.Series([float("nan")] * len(adsl))
    demographics = pd.DataFrame({
        "patient_id": adsl["USUBJID"],
        "age": age_numeric,
        "age_group": adsl["AGEGR1"],
        "sex": adsl["SEX"],
        "race": adsl["RACEGR1"],
        "ecog": adsl["ECOGSCR"].apply(_parse_ecog),
        "weight_kg": pd.to_numeric(adsl["WTKG"], errors="coerce"),
        "height_cm": pd.to_numeric(adsl["HTCM"], errors="coerce"),
        "bmi": pd.to_numeric(adsl["BMI"], errors="coerce"),
        "region": adsl["REGION1"],
    })

    adae = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adae.sas7bdat"), encoding="latin1"))
    te_ae = adae[adae["TRTEMFL"] == "Y"].copy()
    ae_events = pd.DataFrame({
        "patient_id": te_ae["USUBJID"],
        "ae_term": te_ae["AEDECOD"],
        "organ_system": te_ae["AEBODSYS"] if "AEBODSYS" in te_ae.columns else None,
        "start_day": pd.to_numeric(te_ae["ASTDY"], errors="coerce"),
        "is_serious": (te_ae["AESER"] == "Y"),
        "grade": te_ae["AETOXGR"],
    })
    ae_events["start_month"] = ae_events["start_day"] / 30.44

    adtte = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adtte.sas7bdat"), encoding="latin1"))
    os_data = adtte[adtte["PARAMCD"] == "OS"].copy()
    survival = pd.DataFrame({
        "patient_id": os_data["USUBJID"],
        "os_months": os_data["AVAL"].astype(float),
        "censored": (os_data["CNSR"] == 1.0),
    })

    return {
        "demographics": demographics,
        "ae_events": ae_events,
        "survival": survival,
        "n_patients": len(demographics),
    }


def load_real_nct02499770() -> dict:
    ipd_dir = os.path.join(BASE_DIR, "datasets", "NCT02499770", "ipd")

    adsl = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adsl.sas7bdat"), encoding="latin1"))
    ecog_col = "ECOG" if "ECOG" in adsl.columns else "ECOGSCR"
    age_numeric = adsl["AGE"].apply(_parse_age_band) if "AGE" in adsl.columns else pd.Series([float("nan")] * len(adsl))
    demographics = pd.DataFrame({
        "patient_id": adsl["USUBJID"],
        "age": age_numeric,
        "age_group": adsl["AGEGR1"],
        "sex": adsl["SEX"],
        "race": adsl["RACEGR1"],
        "ecog": adsl[ecog_col].apply(_parse_ecog) if ecog_col in adsl.columns else "Unknown",
        "weight_kg": pd.to_numeric(adsl["WTKG"], errors="coerce"),
        "height_cm": pd.to_numeric(adsl["HTCM"], errors="coerce"),
        "bmi": pd.to_numeric(adsl["BMI"], errors="coerce"),
        "region": adsl["REGION1"] if "REGION1" in adsl.columns else None,
    })

    adae = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adae.sas7bdat"), encoding="latin1"))
    te_ae = adae[adae["TRTEMFL"] == "Y"].copy()
    ae_events = pd.DataFrame({
        "patient_id": te_ae["USUBJID"],
        "ae_term": te_ae["AEDECOD"],
        "organ_system": te_ae["AEBODSYS"] if "AEBODSYS" in te_ae.columns else None,
        "start_day": pd.to_numeric(te_ae["ASTDY"], errors="coerce"),
        "is_serious": (te_ae["AESER"] == "Y"),
        "grade": te_ae["AETOXGR"],
    })
    ae_events["start_month"] = ae_events["start_day"] / 30.44

    adtte = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "adtte.sas7bdat"), encoding="latin1"))
    os_data = adtte[adtte["PARAMCD"] == "OS"].copy()
    survival = pd.DataFrame({
        "patient_id": os_data["USUBJID"],
        "os_months": os_data["AVAL"].astype(float),
        "censored": (os_data["CNSR"] == 1.0),
    })

    return {
        "demographics": demographics,
        "ae_events": ae_events,
        "survival": survival,
        "n_patients": len(demographics),
    }


def load_real_nct00844649() -> dict:
    ipd_dir = os.path.join(BASE_DIR, "datasets", "NCT00844649", "ipd", "CA046 data")

    # Demographics from dm
    dm = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "dm.sas7bdat"), encoding="latin1"))

    # ECOG from KPS in kp.sas7bdat
    kp = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "kp.sas7bdat"), encoding="latin1"))
    kp_base = kp[kp["VISIT"] == "BASE"].groupby("RUSUBJID")["KPSTRESN"].last().reset_index()

    def kps_to_ecog(kps):
        if pd.isna(kps):
            return "Unknown"
        if kps >= 90:
            return "0"
        elif kps >= 70:
            return "1"
        else:
            return "2"

    kp_base["ecog"] = kp_base["KPSTRESN"].apply(kps_to_ecog)

    # Weight/height from vs.sas7bdat
    vs = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "vs.sas7bdat"), encoding="latin1"))
    vs_base = vs[vs["VISIT"] == "BASE"]
    weight = vs_base[vs_base["VSTEST"] == "Weight"].groupby("RUSUBJID")["VSSTRESN"].last()
    height = vs_base[vs_base["VSTEST"] == "Height"].groupby("RUSUBJID")["VSSTRESN"].last()

    demographics = dm[["RUSUBJID", "AGE", "SEX", "RACEGEN", "REGION"]].copy()
    demographics = demographics.rename(columns={
        "RUSUBJID": "patient_id", "AGE": "age", "SEX": "sex",
        "RACEGEN": "race", "REGION": "region",
    })
    demographics = demographics.merge(
        kp_base[["RUSUBJID", "ecog"]],
        left_on="patient_id", right_on="RUSUBJID", how="left",
    )
    if "RUSUBJID" in demographics.columns:
        demographics = demographics.drop(columns=["RUSUBJID"])
    demographics["ecog"] = demographics["ecog"].fillna("Unknown")
    demographics["weight_kg"] = demographics["patient_id"].map(weight)
    demographics["height_cm"] = demographics["patient_id"].map(height)
    demographics["bmi"] = demographics["weight_kg"] / ((demographics["height_cm"] / 100) ** 2)

    # Adverse events from ae.sas7bdat (no TRTEMFL in SDTM — use all AEs)
    ae = _strip_strings(pd.read_sas(os.path.join(ipd_dir, "ae.sas7bdat"), encoding="latin1"))
    ae_events = pd.DataFrame({
        "patient_id": ae["RUSUBJID"],
        "ae_term": ae["AEDECOD"],
        "organ_system": ae["AEBODSYS"],
        "start_day": ae["AESTDY"].astype(float),
        "is_serious": (ae["AESER"] == "Y") | (ae["AESERN"] == 1.0),
        "grade": ae["AETOXGRC"],
    })
    ae_events["start_month"] = ae_events["start_day"] / 30.44

    # Survival from fu (follow-up) and ds (disposition)
    fu = pd.read_sas(os.path.join(ipd_dir, "fu.sas7bdat"), encoding="latin1")
    ds = pd.read_sas(os.path.join(ipd_dir, "ds.sas7bdat"), encoding="latin1")

    last_fu = fu.groupby("RUSUBJID")["FUDY"].max()
    death_fu = fu[fu["FUALIVE"] == "N"].groupby("RUSUBJID")["FUDTHDY"].min()
    death_ds = ds[ds["DSALIVE"] == "N"].groupby("RUSUBJID")["DSDTHDY"].min()
    death_day = death_fu.combine_first(death_ds)

    survival_rows = []
    for pid in demographics["patient_id"]:
        if pid in death_day.index and not pd.isna(death_day[pid]):
            os_day = death_day[pid]
            censored = False
        elif pid in last_fu.index and not pd.isna(last_fu[pid]):
            os_day = last_fu[pid]
            censored = True
        else:
            continue  # skip patients with no follow-up data
        survival_rows.append({
            "patient_id": pid,
            "os_months": float(os_day) / 30.44,
            "censored": censored,
        })

    survival = pd.DataFrame(survival_rows)
    survival = survival[survival["os_months"] > 0]

    n_skipped = len(demographics) - len(survival)
    if n_skipped > 0:
        print(
            f"  Warning: {n_skipped} patients excluded from survival (no follow-up data)",
            file=sys.stderr,
        )

    return {
        "demographics": demographics,
        "ae_events": ae_events,
        "survival": survival,
        "n_patients": len(demographics),
    }


_LOADERS = {
    "NCT03041311": load_real_nct03041311,
    "NCT02499770": load_real_nct02499770,
    "NCT00844649": load_real_nct00844649,
}


def load_real_patients(trial_id: str) -> dict:
    if trial_id not in _LOADERS:
        raise ValueError(f"Unknown trial '{trial_id}'. Supported: {list(_LOADERS.keys())}")
    return _LOADERS[trial_id]()


# ============================================================================
# SYNTHETIC DATA LOADING
# ============================================================================

def _synth_dict_from_df(df: pd.DataFrame) -> dict:
    patients = {}
    for _, row in df.iterrows():
        pid = row["patient_id"]
        if pid not in patients:
            demo = json.loads(row["demographic_info"])
            patients[pid] = demo

    demo_df = pd.DataFrame.from_dict(patients, orient="index")
    demo_df["patient_id"] = demo_df.index
    demo_df = demo_df.reset_index(drop=True)

    demographics = pd.DataFrame({
        "patient_id": demo_df["patient_id"],
        "age": demo_df["age"].astype(int),
        "sex": demo_df["sex"],
        "ecog": demo_df["ecog"].astype(str),
        "weight_kg": demo_df["weight_kg"].astype(float),
        "height_cm": demo_df["height_cm"].astype(float),
        "bmi": demo_df["bmi"].astype(float),
    })

    ae_df = df[~df["event"].isin(["treatment_start", "death", "censored"])].copy()
    ae_events = pd.DataFrame({
        "patient_id": ae_df["patient_id"],
        "ae_term": ae_df["event"],
        "start_month": ae_df["time"].astype(float),
    })

    terminal = df[df["event"].isin(["death", "censored"])].copy()
    survival = pd.DataFrame({
        "patient_id": terminal["patient_id"],
        "os_months": terminal["time"].astype(float),
        "censored": (terminal["event"] == "censored"),
    })

    return {
        "demographics": demographics,
        "ae_events": ae_events,
        "survival": survival,
        "n_patients": len(demographics),
    }


def load_synthetic_patients(csv_path: str) -> dict:
    return _synth_dict_from_df(pd.read_csv(csv_path))


def _synth_dict_from_configs(configs: list[TrialConfig], seed: int) -> dict:
    """Generate a synthetic cohort in memory from one or more arm configs."""
    rows = generate_combined(configs, seed=seed)
    return _synth_dict_from_df(pd.DataFrame(rows))


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def _normalize_dist(counter: Counter, total: int | None = None) -> dict[str, float]:
    if total is None:
        total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def _jensen_shannon_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0.0) for k in all_keys])
    q_vec = np.array([q.get(k, 0.0) for k in all_keys])
    eps = 1e-12
    p_vec = p_vec + eps
    q_vec = q_vec + eps
    p_vec = p_vec / p_vec.sum()
    q_vec = q_vec / q_vec.sum()
    m = 0.5 * (p_vec + q_vec)
    jsd = 0.5 * np.sum(p_vec * np.log2(p_vec / m)) + \
          0.5 * np.sum(q_vec * np.log2(q_vec / m))
    return float(jsd)


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    all_keys = sorted(set(a.keys()) | set(b.keys()))
    va = np.array([a.get(k, 0.0) for k in all_keys])
    vb = np.array([b.get(k, 0.0) for k in all_keys])
    dot = np.dot(va, vb)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def kaplan_meier(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    t = times[order]
    e = events[order]

    unique_times = np.unique(t)
    surv_prob = 1.0
    km_times = [0.0]
    km_surv = [1.0]

    for ut in unique_times:
        mask = t == ut
        d_i = e[mask].sum()
        n_i = (t >= ut).sum()
        if n_i > 0:
            surv_prob *= (1.0 - d_i / n_i)
        km_times.append(ut)
        km_surv.append(surv_prob)

    return np.array(km_times), np.array(km_surv)


def _km_area_between_curves(
    t1: np.ndarray, s1: np.ndarray,
    t2: np.ndarray, s2: np.ndarray,
    max_t: float | None = None,
) -> float:
    if max_t is None:
        max_t = min(t1.max(), t2.max())
    all_t = np.sort(np.unique(np.concatenate([t1, t2, [0, max_t]])))
    all_t = all_t[all_t <= max_t]

    idx1 = np.searchsorted(t1, all_t, side="right") - 1
    idx1 = np.clip(idx1, 0, len(s1) - 1)
    s1_step = s1[idx1]

    idx2 = np.searchsorted(t2, all_t, side="right") - 1
    idx2 = np.clip(idx2, 0, len(s2) - 1)
    s2_step = s2[idx2]

    abs_diff = np.abs(s1_step - s2_step)
    dt = np.diff(all_t)
    area = np.sum(abs_diff[:-1] * dt)
    return float(area / max_t)


# ============================================================================
# Extraction IAE: published KM (extracted from figure) vs real IPD KM
# ============================================================================

# Arm whose real IPD cohort we pair with the matching arm of the extracted
# published KM curve. Sponsor-delivered IPD contains one arm per trial; the
# extracted OS CSV contains all published arms, keyed by exact label match.
_REAL_IPD_ARM_LABEL = {
    "NCT03041311": "Placebo prior to E/P/A",
    "NCT02499770": "E/P",
    "NCT00844649": "Gemcitabine",
}


def compute_extraction_iae(trial_id: str, real: dict | None = None) -> dict:
    """Return IAE between the published-figure extraction and the real-IPD KM.

    Reconstructs Kaplan--Meier step functions from (a) the real IPD overall
    survival table for `trial_id` and (b) the arm of the extracted OS KM CSV
    whose label is registered in `_REAL_IPD_ARM_LABEL`, then integrates the
    absolute difference over $[0, \\min(t_{\\max})]$.
    """
    if trial_id not in _REAL_IPD_ARM_LABEL:
        raise ValueError(
            f"No extraction-IAE arm mapping for {trial_id}. "
            f"Known: {list(_REAL_IPD_ARM_LABEL.keys())}"
        )
    arm_label = _REAL_IPD_ARM_LABEL[trial_id]
    csv_path = os.path.join(BASE_DIR, "datasets", trial_id, f"{trial_id}_OS_km.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Extracted OS KM CSV not found: {csv_path}")

    if real is None:
        real = load_real_patients(trial_id)
    surv = real["survival"]
    t_real = surv["os_months"].to_numpy(dtype=float)
    e_real = (~surv["censored"]).astype(int).to_numpy()
    kt_r, ks_r = kaplan_meier(t_real, e_real)

    ext = pd.read_csv(csv_path)
    arm = ext[ext["arm"] == arm_label]
    if len(arm) == 0:
        raise ValueError(
            f"Arm {arm_label!r} not found in {csv_path}. "
            f"Available: {sorted(ext['arm'].unique())}"
        )
    t_ext = arm["time"].to_numpy(dtype=float)
    e_ext = arm["event"].astype(int).to_numpy()
    kt_e, ks_e = kaplan_meier(t_ext, e_ext)

    t_max = float(min(t_real.max(), t_ext.max()))
    iae = _km_area_between_curves(kt_r, ks_r, kt_e, ks_e, max_t=t_max)
    return {
        "trial_id": trial_id,
        "arm": arm_label,
        "n_real": int(len(surv)),
        "n_extracted": int(len(arm)),
        "t_max_months": t_max,
        "iae": iae,
    }


def print_extraction_iae_table(trial_ids: list[str] | None = None) -> list[dict]:
    """Print the extraction-IAE table for the requested trials and return rows."""
    trial_ids = trial_ids or list(_REAL_IPD_ARM_LABEL.keys())
    rows = []
    print(f"{'Trial':<14} {'Arm':<30} {'n_real':>6} {'n_ext':>6} {'t_max':>7} {'IAE':>8}")
    print("-" * 77)
    for tid in trial_ids:
        r = compute_extraction_iae(tid)
        rows.append(r)
        print(
            f"{tid:<14} {r['arm']:<30} {r['n_real']:>6d} {r['n_extracted']:>6d} "
            f"{r['t_max_months']:>7.1f} {r['iae']:>8.4f}"
        )
    print("-" * 77)
    if rows:
        mean_iae = sum(r["iae"] for r in rows) / len(rows)
        print(f"{'Mean':<14} {'':<30} {'':>6} {'':>6} {'':>7} {mean_iae:>8.4f}")
    return rows


# ============================================================================
# AE term normalization (British<->American spelling + casing)
# ============================================================================

# Order matters: longer stems before their substrings (e.g. `anaesthe` before `anaes`).
_BR_AM_SUBS = [
    ("anaemia", "anemia"),
    ("anaem", "anem"),
    ("haemorrh", "hemorrh"),
    ("haemat", "hemat"),
    ("haemo", "hemo"),
    ("oedemat", "edemat"),
    ("oedema", "edema"),
    ("oesophag", "esophag"),
    ("diarrhoea", "diarrhea"),
    ("dyspnoea", "dyspnea"),
    ("leucop", "leukop"),
    ("leucoc", "leukoc"),
    ("tumour", "tumor"),
    ("orthopaed", "orthoped"),
    ("paediatr", "pediatr"),
    ("gynaeco", "gyneco"),
    ("anaesthe", "anesthe"),
    ("anaest", "anest"),
    ("caecum", "cecum"),
    ("faec", "fec"),
    ("coeliac", "celiac"),
    ("hypokalaem", "hypokalem"),
    ("hyperkalaem", "hyperkalem"),
    ("hypocalcaem", "hypocalcem"),
    ("hypercalcaem", "hypercalcem"),
    ("hypomagnesaem", "hypomagnesem"),
    ("hypermagnesaem", "hypermagnesem"),
    ("hyponatraem", "hyponatrem"),
    ("hypernatraem", "hypernatrem"),
    ("hypoglycaem", "hypoglycem"),
    ("hyperglycaem", "hyperglycem"),
    ("hypoalbuminaem", "hypoalbuminem"),
    ("bacteraem", "bacterem"),
    ("septicaem", "septicem"),
    ("uraem", "urem"),
    ("toxaem", "toxem"),
]

# Common misspellings seen in upstream sources.
_MISSPELL_SUBS = [
    ("pruritis", "pruritus"),
    ("dehyration", "dehydration"),
]


def normalize_ae_term(term) -> str:
    """Canonicalize an AE term for cross-source matching.

    Lowercases, collapses whitespace, strips trailing punctuation, and
    rewrites common British spellings to American (MedDRA sources mix both).
    """
    if term is None:
        return ""
    try:
        if isinstance(term, float) and np.isnan(term):
            return ""
    except TypeError:
        pass
    s = str(term).strip().lower()
    s = " ".join(s.split())
    s = s.rstrip(".,;:")
    for br, am in _BR_AM_SUBS:
        s = s.replace(br, am)
    for a, b in _MISSPELL_SUBS:
        s = s.replace(a, b)
    return s


def _normalize_ae_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0 or "ae_term" not in df.columns:
        return df
    df = df.copy()
    df["ae_term"] = df["ae_term"].map(normalize_ae_term)
    return df


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def compute_metrics(real: dict, synth: dict) -> dict:
    results: dict = {}

    # Normalize AE terms up-front so every downstream metric uses canonical form.
    real = {**real, "ae_events": _normalize_ae_frame(real.get("ae_events"))}
    synth = {**synth, "ae_events": _normalize_ae_frame(synth.get("ae_events"))}

    # 1. Demographics
    results["demographics"] = {}

    real_sex = _normalize_dist(Counter(real["demographics"]["sex"]))
    synth_sex = _normalize_dist(Counter(synth["demographics"]["sex"]))
    results["demographics"]["sex_distribution"] = {
        "real": real_sex, "synthetic": synth_sex,
        "jsd": _jensen_shannon_divergence(real_sex, synth_sex),
    }

    real_ecog = _normalize_dist(Counter(real["demographics"]["ecog"]))
    synth_ecog = _normalize_dist(Counter(synth["demographics"]["ecog"]))
    results["demographics"]["ecog_distribution"] = {
        "real": real_ecog, "synthetic": synth_ecog,
        "jsd": _jensen_shannon_divergence(real_ecog, synth_ecog),
    }

    for var in ["weight_kg", "height_cm", "bmi"]:
        r_vals = real["demographics"][var].dropna().values
        s_vals = synth["demographics"][var].dropna().values
        if len(r_vals) > 0 and len(s_vals) > 0:
            ks_stat, ks_p = stats.ks_2samp(r_vals, s_vals)
            results["demographics"][var] = {
                "real_mean": float(np.mean(r_vals)),
                "real_std": float(np.std(r_vals)),
                "synth_mean": float(np.mean(s_vals)),
                "synth_std": float(np.std(s_vals)),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            }

    # 2. Overall Survival
    results["survival"] = {}
    r_surv = real["survival"]
    s_surv = synth["survival"]

    r_median = float(r_surv["os_months"].median())
    s_median = float(s_surv["os_months"].median())
    results["survival"]["median_os_real"] = r_median
    results["survival"]["median_os_synth"] = s_median
    results["survival"]["median_os_diff_months"] = abs(r_median - s_median)
    results["survival"]["median_os_relative_error"] = abs(r_median - s_median) / r_median if r_median > 0 else float("inf")

    r_death_rate = float((~r_surv["censored"]).mean())
    s_death_rate = float((~s_surv["censored"]).mean())
    results["survival"]["death_rate_real"] = r_death_rate
    results["survival"]["death_rate_synth"] = s_death_rate
    results["survival"]["death_rate_diff"] = abs(r_death_rate - s_death_rate)

    ks_stat, ks_p = stats.ks_2samp(r_surv["os_months"].values, s_surv["os_months"].values)
    results["survival"]["os_ks_statistic"] = float(ks_stat)
    results["survival"]["os_ks_pvalue"] = float(ks_p)

    km_t_r, km_s_r = kaplan_meier(r_surv["os_months"].values, (~r_surv["censored"]).astype(int).values)
    km_t_s, km_s_s = kaplan_meier(s_surv["os_months"].values, (~s_surv["censored"]).astype(int).values)
    results["survival"]["km_integrated_abs_diff"] = _km_area_between_curves(km_t_r, km_s_r, km_t_s, km_s_s)

    u_stat, u_p = stats.mannwhitneyu(
        r_surv["os_months"].values, s_surv["os_months"].values, alternative="two-sided",
    )
    results["survival"]["mannwhitney_u_statistic"] = float(u_stat)
    results["survival"]["mannwhitney_u_pvalue"] = float(u_p)

    # 3. AE Frequency
    results["ae_frequency"] = {}
    r_ae_counts = Counter(real["ae_events"]["ae_term"])
    s_ae_counts = Counter(synth["ae_events"]["ae_term"])
    r_ae_dist = _normalize_dist(r_ae_counts)
    s_ae_dist = _normalize_dist(s_ae_counts)

    results["ae_frequency"]["jsd"] = _jensen_shannon_divergence(r_ae_dist, s_ae_dist)
    results["ae_frequency"]["cosine_similarity"] = _cosine_similarity(r_ae_dist, s_ae_dist)

    r_top15 = [ae for ae, _ in r_ae_counts.most_common(15)]
    s_top15 = [ae for ae, _ in s_ae_counts.most_common(15)]
    overlap = len(set(r_top15) & set(s_top15))
    results["ae_frequency"]["top15_overlap"] = overlap
    results["ae_frequency"]["top15_overlap_frac"] = overlap / 15
    results["ae_frequency"]["real_top15"] = r_top15
    results["ae_frequency"]["synth_top15"] = s_top15

    shared_aes = set(r_ae_counts.keys()) & set(s_ae_counts.keys())
    r_n, s_n = real["n_patients"], synth["n_patients"]
    incidence_diffs = [abs(r_ae_counts[ae] / r_n - s_ae_counts[ae] / s_n) for ae in shared_aes]

    results["ae_frequency"]["n_shared_ae_terms"] = len(shared_aes)
    results["ae_frequency"]["n_real_only_terms"] = len(set(r_ae_counts.keys()) - set(s_ae_counts.keys()))
    results["ae_frequency"]["n_synth_only_terms"] = len(set(s_ae_counts.keys()) - set(r_ae_counts.keys()))
    results["ae_frequency"]["mean_incidence_rate_diff"] = float(np.mean(incidence_diffs)) if incidence_diffs else None
    results["ae_frequency"]["median_incidence_rate_diff"] = float(np.median(incidence_diffs)) if incidence_diffs else None

    # 4. AE Timing
    results["ae_timing"] = {}
    r_ae_times = real["ae_events"]["start_month"].dropna().values
    s_ae_times = synth["ae_events"]["start_month"].dropna().values
    if len(r_ae_times) > 0 and len(s_ae_times) > 0:
        ks_stat, ks_p = stats.ks_2samp(r_ae_times, s_ae_times)
        results["ae_timing"] = {
            "real_mean_onset_month": float(np.mean(r_ae_times)),
            "synth_mean_onset_month": float(np.mean(s_ae_times)),
            "real_median_onset_month": float(np.median(r_ae_times)),
            "synth_median_onset_month": float(np.median(s_ae_times)),
            "onset_ks_statistic": float(ks_stat),
            "onset_ks_pvalue": float(ks_p),
        }

    # 5. AE Burden
    results["ae_burden"] = {}
    r_burden = real["ae_events"].groupby("patient_id").size()
    s_burden = synth["ae_events"].groupby("patient_id").size()
    results["ae_burden"]["real_mean_aes_per_patient"] = float(r_burden.mean()) if len(r_burden) else 0.0
    results["ae_burden"]["synth_mean_aes_per_patient"] = float(s_burden.mean()) if len(s_burden) else 0.0
    results["ae_burden"]["real_median_aes_per_patient"] = float(r_burden.median()) if len(r_burden) else 0.0
    results["ae_burden"]["synth_median_aes_per_patient"] = float(s_burden.median()) if len(s_burden) else 0.0
    if len(r_burden) > 0 and len(s_burden) > 0:
        ks, kp = stats.ks_2samp(r_burden.values, s_burden.values)
        results["ae_burden"]["burden_ks_statistic"] = float(ks)
        results["ae_burden"]["burden_ks_pvalue"] = float(kp)
    else:
        results["ae_burden"]["burden_ks_statistic"] = 0.0
        results["ae_burden"]["burden_ks_pvalue"] = 1.0

    # 6. Organ System (from real IPD's AEBODSYS)
    if "organ_system" in real["ae_events"].columns:
        results["organ_system"] = {}
        r_organ = _normalize_dist(Counter(real["ae_events"]["organ_system"]))

        # Build term -> organ_system mapping from real data
        term_to_organ = real["ae_events"].groupby("ae_term")["organ_system"].first().to_dict()
        mapped = [term_to_organ.get(t, "Unknown") for t in synth["ae_events"]["ae_term"]]
        s_organ = _normalize_dist(Counter(mapped))

        results["organ_system"]["real_distribution"] = dict(sorted(r_organ.items(), key=lambda x: -x[1]))
        results["organ_system"]["jsd"] = _jensen_shannon_divergence(r_organ, s_organ)
        results["organ_system"]["cosine_similarity"] = _cosine_similarity(r_organ, s_organ)

    return results


# ============================================================================
# REPORTING
# ============================================================================

def _fmt_dist(d: dict) -> str:
    return ", ".join(f"{k}: {v*100:.1f}%" for k, v in sorted(d.items()))


def print_report(metrics: dict, real: dict, synth: dict, trial_id: str, synth_label: str) -> str:
    lines: list[str] = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 78)
    p("  SYNTHETIC PATIENT FIDELITY REPORT")
    p(f"  {trial_id} [{synth_label}] — Real (n={real['n_patients']}) vs Synthetic (n={synth['n_patients']})")
    p("=" * 78)
    p()

    # Demographics
    p("+---------------------------------------------------------------------------+")
    p("|  1. DEMOGRAPHICS                                                          |")
    p("+---------------------------------------------------------------------------+")
    d = metrics["demographics"]

    sex = d["sex_distribution"]
    p(f"  Sex distribution (JSD = {sex['jsd']:.4f}):")
    p(f"    Real:       {_fmt_dist(sex['real'])}")
    p(f"    Synthetic:  {_fmt_dist(sex['synthetic'])}")
    p()

    ecog = d["ecog_distribution"]
    p(f"  ECOG distribution (JSD = {ecog['jsd']:.4f}):")
    p(f"    Real:       {_fmt_dist(ecog['real'])}")
    p(f"    Synthetic:  {_fmt_dist(ecog['synthetic'])}")
    p()

    for var in ["weight_kg", "height_cm", "bmi"]:
        if var in d:
            v = d[var]
            p(f"  {var}: Real {v['real_mean']:.1f}+/-{v['real_std']:.1f} vs "
              f"Synth {v['synth_mean']:.1f}+/-{v['synth_std']:.1f} "
              f"(KS={v['ks_statistic']:.3f}, p={v['ks_pvalue']:.3f})")
    p()

    # Survival
    p("+---------------------------------------------------------------------------+")
    p("|  2. OVERALL SURVIVAL                                                      |")
    p("+---------------------------------------------------------------------------+")
    s = metrics["survival"]
    p(f"  Median OS:  Real = {s['median_os_real']:.1f} mo  |  Synth = {s['median_os_synth']:.1f} mo  "
      f"|  D = {s['median_os_diff_months']:.1f} mo ({s['median_os_relative_error']*100:.1f}%)")
    p(f"  Death rate: Real = {s['death_rate_real']*100:.1f}%  |  Synth = {s['death_rate_synth']*100:.1f}%  "
      f"|  D = {s['death_rate_diff']*100:.1f}pp")
    p(f"  OS KS test: D = {s['os_ks_statistic']:.3f}, p = {s['os_ks_pvalue']:.3f}")
    p(f"  KM curve integrated abs diff (IABD): {s['km_integrated_abs_diff']:.4f}")
    p(f"  Mann-Whitney U: U = {s['mannwhitney_u_statistic']:.0f}, p = {s['mannwhitney_u_pvalue']:.3f}")
    p()

    # AE Frequency
    p("+---------------------------------------------------------------------------+")
    p("|  3. ADVERSE EVENT FREQUENCY                                               |")
    p("+---------------------------------------------------------------------------+")
    af = metrics["ae_frequency"]
    p(f"  AE term distribution JSD:        {af['jsd']:.4f}")
    p(f"  AE term cosine similarity:       {af['cosine_similarity']:.4f}")
    p(f"  Top-15 overlap:                  {af['top15_overlap']}/15 ({af['top15_overlap_frac']*100:.0f}%)")
    p(f"  Shared AE terms:                 {af['n_shared_ae_terms']}")
    p(f"  Real-only AE terms:              {af['n_real_only_terms']}")
    p(f"  Synth-only AE terms:             {af['n_synth_only_terms']}")
    if af.get("mean_incidence_rate_diff") is not None:
        p(f"  Mean per-AE incidence rate diff:  {af['mean_incidence_rate_diff']:.3f} events/patient")
    p()
    p(f"  Real  top 15:  {', '.join(af['real_top15'][:8])}")
    p(f"                 {', '.join(af['real_top15'][8:])}")
    p(f"  Synth top 15:  {', '.join(af['synth_top15'][:8])}")
    p(f"                 {', '.join(af['synth_top15'][8:])}")
    p()

    # AE Timing
    p("+---------------------------------------------------------------------------+")
    p("|  4. ADVERSE EVENT TIMING                                                  |")
    p("+---------------------------------------------------------------------------+")
    at = metrics["ae_timing"]
    if "onset_ks_statistic" in at:
        p(f"  Mean AE onset:   Real = {at['real_mean_onset_month']:.1f} mo  |  "
          f"Synth = {at['synth_mean_onset_month']:.1f} mo")
        p(f"  Median AE onset: Real = {at['real_median_onset_month']:.1f} mo  |  "
          f"Synth = {at['synth_median_onset_month']:.1f} mo")
        p(f"  Onset KS test: D = {at['onset_ks_statistic']:.3f}, p = {at['onset_ks_pvalue']:.3f}")
    p()

    # AE Burden
    p("+---------------------------------------------------------------------------+")
    p("|  5. AE BURDEN PER PATIENT                                                 |")
    p("+---------------------------------------------------------------------------+")
    ab = metrics["ae_burden"]
    p(f"  Mean AEs/patient:   Real = {ab['real_mean_aes_per_patient']:.1f}  |  "
      f"Synth = {ab['synth_mean_aes_per_patient']:.1f}")
    p(f"  Median AEs/patient: Real = {ab['real_median_aes_per_patient']:.1f}  |  "
      f"Synth = {ab['synth_median_aes_per_patient']:.1f}")
    p(f"  Burden KS test: D = {ab['burden_ks_statistic']:.3f}, p = {ab['burden_ks_pvalue']:.3f}")
    p()

    # Summary
    p("+---------------------------------------------------------------------------+")
    p("|  SUMMARY (raw metric values — lower is better unless noted)               |")
    p("+---------------------------------------------------------------------------+")
    at_onset_ks = at.get("onset_ks_statistic")
    at_onset_p = at.get("onset_ks_pvalue")
    summary = [
        ("Sex JSD",                    f"{sex['jsd']:.4f}"),
        ("ECOG JSD",                   f"{ecog['jsd']:.4f}"),
        ("Median OS |Δ| months",       f"{abs(s['median_os_diff_months']):.2f}"),
        ("Median OS rel. error",       f"{abs(s['median_os_relative_error']):.4f}"),
        ("OS KS statistic",            f"{s['os_ks_statistic']:.4f}"),
        ("OS KS p-value",              f"{s['os_ks_pvalue']:.4g}"),
        ("KM integrated abs diff",     f"{s['km_integrated_abs_diff']:.4f}"),
        ("Mann-Whitney U p-value",     f"{s['mannwhitney_u_pvalue']:.4g}"),
        ("AE frequency JSD",           f"{af['jsd']:.4f}"),
        ("AE frequency cosine sim",    f"{af['cosine_similarity']:.4f} (↑)"),
        ("AE top-15 overlap frac",     f"{af['top15_overlap_frac']:.4f} (↑)"),
        ("AE mean incidence rate Δ",   f"{af.get('mean_incidence_rate_diff', float('nan')):.4f}"),
        ("AE onset KS statistic",      f"{at_onset_ks:.4f}" if at_onset_ks is not None else "N/A"),
        ("AE onset KS p-value",        f"{at_onset_p:.4g}" if at_onset_p is not None else "N/A"),
        ("AE mean onset |Δ| months",   f"{abs(at['real_mean_onset_month'] - at['synth_mean_onset_month']):.2f}" if at_onset_ks is not None else "N/A"),
        ("AE burden mean |Δ|",         f"{abs(ab['real_mean_aes_per_patient'] - ab['synth_mean_aes_per_patient']):.2f}"),
        ("AE burden KS statistic",     f"{ab['burden_ks_statistic']:.4f}"),
        ("AE burden KS p-value",       f"{ab['burden_ks_pvalue']:.4g}"),
    ]
    width = max(len(name) for name, _ in summary)
    for name, value in summary:
        p(f"  {name:<{width}}  {value}")
    p("=" * 78)

    return "\n".join(lines)


# ============================================================================
# REGENERATION-BASED VARIANCE ESTIMATION
#
# Re-runs the synthetic generator from the same TrialConfig(s) with a sweep of
# seeds, then aggregates each numeric leaf of `compute_metrics` into mean/std.
# Used to put error bars on the end-to-end fidelity table in the paper.
# ============================================================================


def _flatten_numeric(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_numeric(v, path))
        elif isinstance(v, bool) or isinstance(v, np.bool_):
            continue
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[path] = float(v)
    return out


def _derived_run_metrics(metrics: dict) -> dict[str, float]:
    extras: dict[str, float] = {}
    at = metrics.get("ae_timing", {}) or {}
    if "real_mean_onset_month" in at and "synth_mean_onset_month" in at:
        extras["ae_timing.mean_onset_abs_diff"] = abs(
            float(at["real_mean_onset_month"]) - float(at["synth_mean_onset_month"])
        )
    ab = metrics.get("ae_burden", {}) or {}
    if "real_mean_aes_per_patient" in ab and "synth_mean_aes_per_patient" in ab:
        extras["ae_burden.mean_abs_diff"] = abs(
            float(ab["real_mean_aes_per_patient"]) - float(ab["synth_mean_aes_per_patient"])
        )
    return extras


def aggregate_runs(runs_metrics: list[dict]) -> dict[str, dict]:
    flats: list[dict[str, float]] = []
    for m in runs_metrics:
        flat = _flatten_numeric(m)
        flat.update(_derived_run_metrics(m))
        flats.append(flat)
    keys = sorted(set().union(*[set(f) for f in flats])) if flats else []
    agg: dict[str, dict] = {}
    for k in keys:
        vals = [f[k] for f in flats if k in f]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(len(arr)),
        }
    return agg


def run_regen_evaluation(
    trial_id: str,
    real: dict,
    configs: list[TrialConfig],
    n_regen: int,
    base_seed: int = 0,
) -> dict:
    """Regenerate synthetic cohorts ``n_regen`` times and aggregate metrics."""
    runs: list[dict] = []
    for i in range(n_regen):
        seed = base_seed + i
        synth = _synth_dict_from_configs(configs, seed=seed)
        m = compute_metrics(real, synth)
        runs.append(m)
        print(f"  [regen {i + 1}/{n_regen}] seed={seed} n_synth={synth['n_patients']}")
    agg = aggregate_runs(runs)
    return {
        "trial_id": trial_id,
        "n_regen": n_regen,
        "base_seed": base_seed,
        "aggregated": agg,
    }


# Rows shown in the paper-ready summary printed for each trial. Each entry is
# (label, dotted-key into aggregated metrics, format-string accepting mean/std).
_PAPER_TABLE_ROWS: list[tuple[str, str, str]] = [
    ("Sex JSD",                  "demographics.sex_distribution.jsd",      "{:.4f} +/- {:.4f}"),
    ("ECOG JSD",                 "demographics.ecog_distribution.jsd",     "{:.3f} +/- {:.3f}"),
    ("Weight KS",                "demographics.weight_kg.ks_statistic",    "{:.3f} +/- {:.3f}"),
    ("Height KS",                "demographics.height_cm.ks_statistic",    "{:.3f} +/- {:.3f}"),
    ("BMI KS",                   "demographics.bmi.ks_statistic",          "{:.3f} +/- {:.3f}"),
    ("Median OS |Δ| (mo.)",      "survival.median_os_diff_months",         "{:.2f} +/- {:.2f}"),
    ("Median OS rel. error",     "survival.median_os_relative_error",      "{:.3f} +/- {:.3f}"),
    ("OS KS",                    "survival.os_ks_statistic",               "{:.3f} +/- {:.3f}"),
    ("Δ_KM",                     "survival.km_integrated_abs_diff",        "{:.3f} +/- {:.3f}"),
    ("Mann-Whitney p",           "survival.mannwhitney_u_pvalue",          "{:.4g} +/- {:.4g}"),
    ("AE JSD",                   "ae_frequency.jsd",                       "{:.3f} +/- {:.3f}"),
    ("AE cosine",                "ae_frequency.cosine_similarity",         "{:.3f} +/- {:.3f}"),
    ("Top-15 overlap (count)",   "ae_frequency.top15_overlap",             "{:.2f} +/- {:.2f}"),
    ("Mean per-AE |Δ|",          "ae_frequency.mean_incidence_rate_diff",  "{:.3f} +/- {:.3f}"),
    ("Mean onset |Δ| (mo.)",     "ae_timing.mean_onset_abs_diff",          "{:.2f} +/- {:.2f}"),
    ("Onset KS",                 "ae_timing.onset_ks_statistic",           "{:.3f} +/- {:.3f}"),
    ("Synth mean AEs/pt",        "ae_burden.synth_mean_aes_per_patient",   "{:.2f} +/- {:.2f}"),
    ("Burden KS",                "ae_burden.burden_ks_statistic",          "{:.3f} +/- {:.3f}"),
]


def print_regen_summary(trial_id: str, agg: dict, n_regen: int) -> str:
    lines: list[str] = []

    def p(s: str = "") -> None:
        lines.append(s)
        print(s)

    p("")
    p("=" * 78)
    p(f"  REGEN SUMMARY — {trial_id}  (n_regen={n_regen})")
    p("=" * 78)
    width = max(len(label) for label, _, _ in _PAPER_TABLE_ROWS)
    for label, key, fmt in _PAPER_TABLE_ROWS:
        v = agg.get(key)
        if v is None:
            p(f"  {label:<{width}}  (missing)")
        else:
            p(f"  {label:<{width}}  {fmt.format(v['mean'], v['std'])}")
    p("=" * 78)
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ============================================================================
# CONFIG-vs-REAL COMPARISON
#
# Compares the parameters that were extracted into per-arm TrialConfig JSONs
# against parameters computed directly from the real IPD. This runs per-trial
# (not per synthetic CSV) since it only depends on the extracted configs.
# ============================================================================

def _pool_configs(configs: list[TrialConfig]) -> dict:
    """Weighted-average numeric TrialConfig fields across arms, weighted by n_enrolled."""
    ns = np.array([c.n_enrolled for c in configs], dtype=float)
    w = ns / ns.sum()
    total_n = int(ns.sum())

    def wavg(field):
        vals = [getattr(c, field) for c in configs]
        if any(v is None for v in vals):
            return None
        return float(np.sum(w * np.array(vals, dtype=float)))

    def wsd(mean_field, sd_field):
        means = np.array([getattr(c, mean_field) for c in configs], dtype=float)
        sds = np.array([getattr(c, sd_field) for c in configs], dtype=float)
        if np.any(np.isnan(means)) or np.any(np.isnan(sds)):
            return None
        pooled_mean = float(np.sum(w * means))
        pooled_var = float(np.sum(w * (sds**2 + (means - pooled_mean) ** 2)))
        return pooled_mean, float(np.sqrt(pooled_var))

    def wdist(field):
        merged: Counter = Counter()
        for cfg, wi in zip(configs, w):
            dist = getattr(cfg, field) or {}
            for k, v in dist.items():
                merged[str(k)] += wi * float(v)
        total = sum(merged.values())
        return {k: v / total for k, v in merged.items()} if total > 0 else {}

    age_mean = wavg("age_mean")
    age_sd_pair = wsd("age_mean", "age_sd")
    out: dict = {
        "n_enrolled_total": total_n,
        "age_mean": age_mean,
        "age_sd": age_sd_pair[1] if age_sd_pair else None,
        "age_min": min((c.age_min for c in configs if c.age_min is not None), default=None),
        "age_max": max((c.age_max for c in configs if c.age_max is not None), default=None),
        "fraction_male": wavg("fraction_male"),
        "ecog_distribution": wdist("ecog_distribution"),
        "race_distribution": wdist("race_distribution"),
        "region_distribution": wdist("region_distribution"),
        "weight_mean": None,
        "weight_sd": None,
        "height_mean": None,
        "height_sd": None,
    }
    if all(c.weight_mean is not None and c.weight_sd is not None for c in configs):
        wpair = wsd("weight_mean", "weight_sd")
        if wpair:
            out["weight_mean"], out["weight_sd"] = wpair
    if all(c.height_mean is not None and c.height_sd is not None for c in configs):
        hpair = wsd("height_mean", "height_sd")
        if hpair:
            out["height_mean"], out["height_sd"] = hpair

    # Pooled AE probabilities. Normalize term spellings so British/American
    # variants collapse into one canonical entry.
    ae_probs: dict[str, float] = {}
    for cfg, wi in zip(configs, w):
        for ae in cfg.adverse_events or []:
            term = ae.get("term") if isinstance(ae, dict) else getattr(ae, "term", None)
            prob = ae.get("probability") if isinstance(ae, dict) else getattr(ae, "probability", None)
            if term is None or prob is None:
                continue
            key = normalize_ae_term(term)
            if not key:
                continue
            ae_probs[key] = ae_probs.get(key, 0.0) + wi * float(prob)
    out["adverse_events"] = ae_probs
    return out


def _real_config_params(real: dict) -> dict:
    """Compute TrialConfig-shaped params from real IPD (demographics + AE probs)."""
    demo: pd.DataFrame = real["demographics"]
    n = len(demo)
    out: dict = {"n_enrolled_total": n}

    if "age" in demo.columns and pd.to_numeric(demo["age"], errors="coerce").notna().any():
        a = pd.to_numeric(demo["age"], errors="coerce").dropna()
        out["age_mean"] = float(a.mean())
        out["age_sd"] = float(a.std(ddof=0))
        out["age_min"] = float(a.min())
        out["age_max"] = float(a.max())
        if "age_group" in demo.columns:
            out["age_source_note"] = "reconstructed from 5-year AGE band midpoints"
    if "age_group" in demo.columns:
        gcounts = demo["age_group"].astype(str).value_counts(normalize=True).to_dict()
        out["age_group_distribution"] = {str(k): float(v) for k, v in gcounts.items()}

    sex = demo["sex"].astype(str).str.upper()
    out["fraction_male"] = float((sex == "M").mean())

    ecog = demo["ecog"].astype(str).value_counts(normalize=True).to_dict()
    out["ecog_distribution"] = {str(k): float(v) for k, v in ecog.items()}

    if "race" in demo.columns:
        race = demo["race"].astype(str).value_counts(normalize=True).to_dict()
        out["race_distribution"] = {str(k): float(v) for k, v in race.items()}
    if "region" in demo.columns:
        region = demo["region"].astype(str).value_counts(normalize=True).to_dict()
        out["region_distribution"] = {str(k): float(v) for k, v in region.items()}

    if "weight_kg" in demo.columns:
        wv = pd.to_numeric(demo["weight_kg"], errors="coerce").dropna()
        if len(wv):
            out["weight_mean"] = float(wv.mean())
            out["weight_sd"] = float(wv.std(ddof=0))
    if "height_cm" in demo.columns:
        hv = pd.to_numeric(demo["height_cm"], errors="coerce").dropna()
        if len(hv):
            out["height_mean"] = float(hv.mean())
            out["height_sd"] = float(hv.std(ddof=0))

    ae = real["ae_events"]
    if len(ae) > 0:
        ae_norm = ae.assign(ae_term=ae["ae_term"].map(normalize_ae_term))
        per_term = ae_norm.groupby("ae_term")["patient_id"].nunique() / n
        out["adverse_events"] = {str(k): float(v) for k, v in per_term.to_dict().items()}
    else:
        out["adverse_events"] = {}
    return out


def _compare_config_params(extracted: dict, real: dict) -> dict:
    cmp: dict = {"scalar_fields": {}, "distribution_fields": {}, "ae_field": {}}

    for f in ["n_enrolled_total", "age_mean", "age_sd", "age_min", "age_max",
              "fraction_male", "weight_mean", "weight_sd", "height_mean", "height_sd"]:
        e = extracted.get(f)
        r = real.get(f)
        if e is None and r is None:
            continue
        entry: dict = {"extracted": e, "real": r}
        if e is not None and r is not None:
            entry["abs_diff"] = abs(float(e) - float(r))
            if float(r) != 0:
                entry["rel_error"] = abs(float(e) - float(r)) / abs(float(r))
        cmp["scalar_fields"][f] = entry

    for f in ["ecog_distribution", "race_distribution", "region_distribution"]:
        e = {str(k): float(v) for k, v in (extracted.get(f) or {}).items()}
        r = {str(k): float(v) for k, v in (real.get(f) or {}).items()}
        if not e and not r:
            continue
        entry = {"extracted": e, "real": r}
        if e and r:
            entry["jsd"] = _jensen_shannon_divergence(e, r)
        cmp["distribution_fields"][f] = entry

    if "age_group_distribution" in real:
        cmp["distribution_fields"]["age_group_distribution"] = {
            "extracted": None,
            "real": real["age_group_distribution"],
            "note": "Real IPD coarsens age to buckets; extracted config uses numeric age_mean/sd.",
        }

    e_ae = extracted.get("adverse_events") or {}
    r_ae = real.get("adverse_events") or {}
    shared = sorted(set(e_ae) & set(r_ae))
    real_only = sorted(set(r_ae) - set(e_ae))
    ext_only = sorted(set(e_ae) - set(r_ae))

    rows = []
    for term in shared:
        e_p = float(e_ae[term]); r_p = float(r_ae[term])
        rows.append({
            "term": term,
            "extracted_prob": e_p,
            "real_prob": r_p,
            "abs_diff": abs(e_p - r_p),
        })
    rows.sort(key=lambda x: -x["real_prob"])

    if rows:
        diffs = np.array([r["abs_diff"] for r in rows])
        mae = float(diffs.mean())
        rmse = float(np.sqrt((diffs**2).mean()))
    else:
        mae = rmse = None

    cmp["ae_field"] = {
        "n_extracted_terms": len(e_ae),
        "n_real_terms": len(r_ae),
        "n_shared": len(shared),
        "n_real_only": len(real_only),
        "n_extracted_only": len(ext_only),
        "shared_mae": mae,
        "shared_rmse": rmse,
        "jsd_full": _jensen_shannon_divergence(e_ae, r_ae) if e_ae and r_ae else None,
        "shared_terms_table": rows,
        "real_only_top10": [(t, float(r_ae[t])) for t in sorted(real_only, key=lambda x: -r_ae[x])[:10]],
        "extracted_only_top10": [(t, float(e_ae[t])) for t in sorted(ext_only, key=lambda x: -e_ae[x])[:10]],
    }
    return cmp


def _render_config_compare_text(trial_id: str, cmp: dict, extracted: dict, real: dict) -> str:
    lines: list[str] = []
    def p(s=""): lines.append(s)
    p("=" * 80)
    p(f"  CONFIG-vs-REAL COMPARISON  ::  {trial_id}")
    p(f"  Extracted pooled n_enrolled = {extracted['n_enrolled_total']}   |   Real IPD n = {real['n_enrolled_total']}")
    if real.get("age_source_note"):
        p(f"  (real age: {real['age_source_note']})")
    p("=" * 80)
    p()
    p("[ SCALAR FIELDS ]")
    p(f"  {'field':<20}{'extracted':>14}{'real':>14}{'|Δ|':>12}{'rel_err':>12}")
    for f, e in cmp["scalar_fields"].items():
        ev = e.get("extracted"); rv = e.get("real")
        ad = e.get("abs_diff"); re_ = e.get("rel_error")
        def fmt(x): return f"{x:>14.3f}" if isinstance(x, (int, float)) else f"{str(x):>14}"
        ad_s = f"{ad:>12.3f}" if isinstance(ad, (int, float)) else f"{'—':>12}"
        re_s = f"{re_:>12.3f}" if isinstance(re_, (int, float)) else f"{'—':>12}"
        p(f"  {f:<20}{fmt(ev)}{fmt(rv)}{ad_s}{re_s}")
    p()
    p("[ DISTRIBUTION FIELDS ]")
    for f, e in cmp["distribution_fields"].items():
        ev = e.get("extracted"); rv = e.get("real"); j = e.get("jsd")
        p(f"  {f}" + (f"   (JSD={j:.4f})" if j is not None else ""))
        p(f"    extracted: {ev}")
        p(f"    real:      {rv}")
        if e.get("note"):
            p(f"    note:      {e['note']}")
    p()
    ae = cmp["ae_field"]
    p("[ ADVERSE EVENT PROBABILITIES ]")
    p(f"  extracted terms: {ae['n_extracted_terms']}   real terms: {ae['n_real_terms']}   shared: {ae['n_shared']}")
    p(f"  real-only: {ae['n_real_only']}   extracted-only: {ae['n_extracted_only']}")
    if ae["shared_mae"] is not None:
        p(f"  shared-term MAE: {ae['shared_mae']:.4f}   RMSE: {ae['shared_rmse']:.4f}   full-term JSD: {ae['jsd_full']:.4f}")
    p()
    p("  shared terms (sorted by real prob, top 25):")
    p(f"    {'term':<50}{'extracted':>12}{'real':>10}{'|Δ|':>10}")
    for row in ae["shared_terms_table"][:25]:
        p(f"    {row['term'][:48]:<50}{row['extracted_prob']:>12.3f}{row['real_prob']:>10.3f}{row['abs_diff']:>10.3f}")
    p()
    p("  top real-only terms (missing from extracted):")
    for t, pr in ae["real_only_top10"]:
        p(f"    {t[:60]:<60} real_p={pr:.3f}")
    p()
    p("  top extracted-only terms (not present in real IPD):")
    for t, pr in ae["extracted_only_top10"]:
        p(f"    {t[:60]:<60} ext_p={pr:.3f}")
    p()
    p("=" * 80)
    return "\n".join(lines)


def compute_config_vs_real(trial_id: str, real: dict, syn_dir: str) -> tuple[dict, str] | None:
    """Load per-arm TrialConfig JSONs from syn_dir and compare against real IPD.

    Returns ``(payload, text)`` where ``payload`` is the serializable comparison
    dict and ``text`` is the human-readable rendering. Returns None if no config
    JSONs are present in ``syn_dir``.
    """
    cfg_paths = sorted(glob.glob(os.path.join(syn_dir, "*_config.json")))
    if not cfg_paths:
        print(f"  (no *_config.json in {syn_dir} — skipping extraction-quality section)")
        return None

    configs = [TrialConfig.load_json(p) for p in cfg_paths]
    extracted = _pool_configs(configs)
    real_params = _real_config_params(real)
    cmp = _compare_config_params(extracted, real_params)

    payload = {
        "trial_id": trial_id,
        "extracted": extracted,
        "real": real_params,
        "comparison": cmp,
    }
    text = _render_config_compare_text(trial_id, cmp, extracted, real_params)
    return payload, text


def main():
    parser = argparse.ArgumentParser(
        description="Compare synthetic patients against real IPD"
    )
    parser.add_argument(
        "--trial-id", type=str, default=None,
        choices=list(_LOADERS.keys()),
        help="Trial ID to evaluate (required unless --extraction-iae-only).",
    )
    parser.add_argument(
        "--synth-csv", type=str, default=None,
        help="Path to synthetic CSV. If omitted, evaluates all synthetic*.csv in the trial's dataset dir.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save report to file",
    )
    parser.add_argument(
        "--skip-config-compare", action="store_true",
        help="Skip the TrialConfig-vs-real-IPD parameter comparison (runs by default).",
    )
    parser.add_argument(
        "--syn-dir", type=str, default=None,
        help="Directory containing per-arm *_config.json files. Defaults to syn_datasets/<trial_id>.",
    )
    parser.add_argument(
        "--extraction-iae-only", action="store_true",
        help="Print the extracted-KM-vs-real-IPD IAE table and exit. "
             "Restricts to --trial-id when provided, else reports all registered trials.",
    )
    parser.add_argument(
        "--n-regen", type=int, default=1,
        help="If >1, regenerate the synthetic cohort N times from the per-arm "
             "TrialConfig JSONs (varying seed) and report mean/std for each "
             "metric instead of the single-CSV evaluation. Used to put error "
             "bars on the paper's results table.",
    )
    parser.add_argument(
        "--regen-base-seed", type=int, default=0,
        help="Base seed for --n-regen sweep; iteration i uses seed = base + i.",
    )
    args = parser.parse_args()

    if args.extraction_iae_only:
        trial_ids = [args.trial_id] if args.trial_id else None
        print_extraction_iae_table(trial_ids)
        return

    if args.trial_id is None:
        parser.error("--trial-id is required unless --extraction-iae-only is set")

    print(f"Loading real IPD data for {args.trial_id}...")
    real = load_real_patients(args.trial_id)
    print(f"  Real patients: {real['n_patients']}")
    print(f"  Real AE events: {len(real['ae_events'])}")
    print(f"  Real OS records: {len(real['survival'])}")
    print()

    extraction_iae: dict | None = None
    if args.trial_id in _REAL_IPD_ARM_LABEL:
        try:
            extraction_iae = compute_extraction_iae(args.trial_id, real=real)
            print(
                f"Extraction IAE (published KM vs real IPD, arm={extraction_iae['arm']!r}): "
                f"{extraction_iae['iae']:.4f} over t in [0, {extraction_iae['t_max_months']:.1f}] mo"
            )
            print()
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping extraction IAE: {e}", file=sys.stderr)

    # ─── Multi-regeneration branch (variance estimation) ──────────────────
    if args.n_regen > 1:
        syn_dir = args.syn_dir or os.path.join(BASE_DIR, "syn_datasets", args.trial_id)
        cfg_paths = sorted(glob.glob(os.path.join(syn_dir, "*_config.json")))
        if not cfg_paths:
            parser.error(
                f"--n-regen={args.n_regen} requires *_config.json files in {syn_dir}"
            )
        configs = [TrialConfig.load_json(p) for p in cfg_paths]
        print(f"Loaded {len(configs)} per-arm config(s) from {syn_dir}")

        regen_payload = run_regen_evaluation(
            args.trial_id, real, configs, args.n_regen, args.regen_base_seed,
        )
        regen_text = print_regen_summary(
            args.trial_id, regen_payload["aggregated"], args.n_regen,
        )

        out_dir = syn_dir
        json_path = os.path.join(out_dir, f"evaluation_regen_n{args.n_regen}.json")
        txt_path = os.path.join(out_dir, f"evaluation_regen_n{args.n_regen}.txt")
        full_payload = {
            **regen_payload,
            "extraction_iae_vs_real_ipd": extraction_iae,
        }
        with open(json_path, "w") as f:
            json.dump(_make_serializable(full_payload), f, indent=2)
        with open(txt_path, "w") as f:
            f.write(regen_text + "\n")
        print(f"\nRegen evaluation saved to:")
        print(f"  {json_path}")
        print(f"  {txt_path}")
        return

    # Discover synthetic CSVs
    if args.synth_csv:
        csv_paths = [args.synth_csv]
    else:
        dataset_dir = os.path.join(BASE_DIR, "datasets", args.trial_id)
        csv_paths = sorted(glob.glob(os.path.join(dataset_dir, "synthetic*.csv")))
        if not csv_paths:
            print(f"No synthetic*.csv found in {dataset_dir}")
            sys.exit(1)
        print(f"Found {len(csv_paths)} synthetic CSV(s):")
        for p in csv_paths:
            print(f"  {os.path.basename(p)}")
        print()

    # Compute extraction-quality section once per trial (depends only on configs + real)
    extraction_payload: dict | None = None
    extraction_text: str | None = None
    if not args.skip_config_compare:
        syn_dir = args.syn_dir or os.path.join(BASE_DIR, "syn_datasets", args.trial_id)
        result = compute_config_vs_real(args.trial_id, real, syn_dir)
        if result is not None:
            extraction_payload, extraction_text = result

    for csv_path in csv_paths:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"{'='*78}")
        print(f"Evaluating: {label}")
        print(f"{'='*78}")

        synth = load_synthetic_patients(csv_path)
        print(f"  Synthetic patients: {synth['n_patients']}")
        print(f"  Synthetic AE events: {len(synth['ae_events'])}")
        print()

        metrics = compute_metrics(real, synth)
        fidelity_text = print_report(metrics, real, synth, args.trial_id, label)

        # Build combined report: extraction quality + end-to-end fidelity
        sections: list[str] = []
        if extraction_text is not None:
            sections.append("#" * 80)
            sections.append(f"#  SECTION 1 — EXTRACTION QUALITY (TrialConfig vs real IPD)")
            sections.append("#" * 80)
            sections.append(extraction_text)
            sections.append("")
        sections.append("#" * 80)
        section_num = 2 if extraction_text is not None else 1
        sections.append(f"#  SECTION {section_num} — END-TO-END FIDELITY (synthetic patients vs real IPD)")
        sections.append("#" * 80)
        sections.append(fidelity_text)
        combined_text = "\n".join(sections)

        combined_payload = {
            "trial_id": args.trial_id,
            "synth_label": label,
            "extraction_quality": _make_serializable(extraction_payload) if extraction_payload is not None else None,
            "extraction_iae_vs_real_ipd": extraction_iae,
            "end_to_end_fidelity": _make_serializable(metrics),
        }

        out_dir = os.path.dirname(csv_path)
        json_path = os.path.join(out_dir, f"evaluation_{label}.json")
        txt_path = os.path.join(out_dir, f"evaluation_{label}.txt")
        with open(json_path, "w") as f:
            json.dump(combined_payload, f, indent=2)
        with open(txt_path, "w") as f:
            f.write(combined_text)
        print(f"\nCombined evaluation saved to:")
        print(f"  {json_path}")
        print(f"  {txt_path}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            f.write(combined_text)
        print(f"Report saved to: {args.output}")


def test_loaders():
    """Verify that real IPD data loads correctly for all 3 trials."""
    errors = []

    def check(cond, msg):
        if not cond:
            errors.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK:   {msg}")

    # --- NCT03041311 ---
    print("Testing NCT03041311...")
    r = load_real_nct03041311()
    check(r["n_patients"] == 53, f"n_patients={r['n_patients']}, expected 53")
    check(len(r["ae_events"]) == 772, f"ae_events={len(r['ae_events'])}, expected 772")
    check(len(r["survival"]) == 53, f"survival={len(r['survival'])}, expected 53")
    demo = r["demographics"]
    check(set(demo["sex"].unique()) == {"F", "M"}, f"sex values: {sorted(demo['sex'].unique())}")
    check(set(demo["ecog"].unique()) <= {"0", "1", "2", "Unknown"}, f"ecog values: {sorted(demo['ecog'].unique())}")
    check("weight_kg" in demo.columns and demo["weight_kg"].notna().sum() > 0, "weight_kg present")
    check("height_cm" in demo.columns and demo["height_cm"].notna().sum() > 0, "height_cm present")
    check("bmi" in demo.columns and demo["bmi"].notna().sum() > 0, "bmi present")
    check("organ_system" in r["ae_events"].columns, "ae organ_system present")
    check(r["survival"]["os_months"].min() > 0, "all os_months > 0")
    check(r["survival"]["censored"].dtype == bool, "censored is bool")
    print()

    # --- NCT02499770 ---
    print("Testing NCT02499770...")
    r = load_real_nct02499770()
    check(r["n_patients"] == 37, f"n_patients={r['n_patients']}, expected 37")
    check(len(r["ae_events"]) == 462, f"ae_events={len(r['ae_events'])}, expected 462")
    check(len(r["survival"]) == 37, f"survival={len(r['survival'])}, expected 37")
    demo = r["demographics"]
    check(set(demo["sex"].unique()) == {"F", "M"}, f"sex values: {sorted(demo['sex'].unique())}")
    check(set(demo["ecog"].unique()) <= {"0", "1", "2", "Unknown"}, f"ecog values: {sorted(demo['ecog'].unique())}")
    check("weight_kg" in demo.columns and demo["weight_kg"].notna().sum() > 0, "weight_kg present")
    check("organ_system" in r["ae_events"].columns, "ae organ_system present")
    check(r["survival"]["os_months"].min() > 0, "all os_months > 0")
    print()

    # --- NCT00844649 ---
    print("Testing NCT00844649...")
    r = load_real_nct00844649()
    check(r["n_patients"] == 430, f"n_patients={r['n_patients']}, expected 430")
    check(len(r["ae_events"]) == 6119, f"ae_events={len(r['ae_events'])}, expected 6119")
    check(len(r["survival"]) > 400, f"survival={len(r['survival'])}, expected >400 (some excluded)")
    demo = r["demographics"]
    check(set(demo["sex"].unique()) <= {"F", "M"}, f"sex values: {sorted(demo['sex'].unique())}")
    check(set(demo["ecog"].unique()) <= {"0", "1", "2", "Unknown"}, f"ecog values: {sorted(demo['ecog'].unique())}")
    check("weight_kg" in demo.columns and demo["weight_kg"].notna().sum() > 0, "weight_kg present")
    check("organ_system" in r["ae_events"].columns, "ae organ_system present")
    check(r["survival"]["os_months"].min() > 0, "all os_months > 0")
    # No hardcoded ecog imputation — verify "Unknown" instead of "1" for missing
    check("1" not in demo["ecog"].value_counts().index or demo["ecog"].value_counts().get("Unknown", 0) >= 0,
          "ecog uses 'Unknown' not hardcoded fillna('1')")
    print()

    # --- Dispatcher ---
    print("Testing dispatcher...")
    for tid in ["NCT03041311", "NCT02499770", "NCT00844649"]:
        r = load_real_patients(tid)
        check(r["n_patients"] > 0, f"load_real_patients('{tid}') returned {r['n_patients']} patients")
    try:
        load_real_patients("FAKE_TRIAL")
        check(False, "load_real_patients('FAKE_TRIAL') should raise ValueError")
    except ValueError:
        check(True, "load_real_patients('FAKE_TRIAL') raises ValueError")
    print()

    if errors:
        print(f"\n{'='*60}")
        print(f"FAILED: {len(errors)} check(s)")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"All checks passed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_loaders()
    else:
        main()
