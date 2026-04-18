"""
Synthetic Patient Generator — Generic Engine
=============================================

Generates synthetic patient-level adverse event data from any clinical trial's
aggregate results. Configurable via `TrialConfig` dataclass.

Pipeline:
    1. TrialConfig defines: adverse events, baseline demographics, survival
    2. TrialConfig.build_subcategories() auto-generates patient archetypes
    3. Per-patient sampling: Weibull survival + cycle-based AE occurrence
    4. Output: tabular (patient_id, time, event, demographic_info)

Example usage:

python synthetic_patient_generator.py arm_control.json arm_experimental.json -n 200 -o combined.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np


# ============================================================================
# RISK PROFILE — Configurable patient-factor risk modifiers
# ============================================================================

# Canonical short→full MedDRA SOC name mapping for backward compat
_ORGAN_SHORT_TO_FULL = {
    "General disorders": "General disorders and administration site conditions",
    "Respiratory disorders": "Respiratory, thoracic and mediastinal disorders",
    "Musculoskeletal disorders": "Musculoskeletal and connective tissue disorders",
}


def _normalize_organ(name: str) -> str:
    """Normalize an organ system name to full MedDRA SOC form."""
    return _ORGAN_SHORT_TO_FULL.get(name, name)


# Evidence-based default risk modifiers by patient factor.
# Sources cited inline; see citations.bib for full references.
# Multipliers are relative (1.0 = baseline).
# Keys use full MedDRA System Organ Class (SOC) names.
_DEFAULT_AE_RISK_MODIFIERS: dict[str, dict[str, float]] = {
    # Older patients (≥65): higher hematologic/cardiac/infection/renal risk.
    # Directional support: Hurria et al. JCO 2011 (OR 1.85 overall grade 3-5
    # toxicity for age ≥72) [hurria2011]; Extermann et al. Cancer 2012 (CRASH
    # score) [extermann2012]; Kuderer et al. Cancer 2006 (FN mortality by
    # comorbidity) [kuderer2006]; Launay-Vacher et al. Ann Oncol 2007 (renal
    # insufficiency prevalence) [launay2007].
    # Per-SOC multipliers are clinical estimates; no per-organ RRs from these
    # papers.
    "older": {
        "Blood and lymphatic system disorders": 1.5,
        "Cardiac disorders": 1.8,
        "Infections and infestations": 1.6,
        "Nervous system disorders": 1.3,
        "Vascular disorders": 1.4,
        "Metabolism and nutrition disorders": 1.2,
        "Respiratory, thoracic and mediastinal disorders": 1.2,
        "Musculoskeletal and connective tissue disorders": 1.2,
        "Renal and urinary disorders": 1.4,
    },
    "younger": {
        "Blood and lymphatic system disorders": 0.75,  # inverse of older
        "Cardiac disorders": 0.55,
        "Infections and infestations": 0.80,
        "Gastrointestinal disorders": 1.05,
        "Skin and subcutaneous tissue disorders": 1.1,
    },
    # Female: higher overall chemo toxicity (OR 1.34, 95% CI 1.27-1.42 for
    # grade 3+ AEs; hematologic OR 1.30, 95% CI 1.23-1.37; symptomatic
    # OR 1.33, 95% CI 1.26-1.41) [unger2022].
    # Directional support: Schmetzer & Flörcken, Handb Exp Pharmacol 2012
    # (review of sex-based PK/toxicity differences) [schmetzer2012].
    # Per-SOC multipliers are clinical estimates informed by these overall ORs.
    "female": {
        "Gastrointestinal disorders": 1.3,
        "Blood and lymphatic system disorders": 1.25,
        "Nervous system disorders": 1.2,
        "Endocrine disorders": 1.2,
        "Immune system disorders": 1.15,
        "Skin and subcutaneous tissue disorders": 1.15,
        "Cardiac disorders": 0.85,
    },
    "male": {
        "Cardiac disorders": 1.2,
        "Vascular disorders": 1.1,
        "Renal and urinary disorders": 1.1,
    },
    # Poor performance status (ECOG ≥2): higher AE risk.
    # Directional support: Sargent et al. JCO 2009 (PS 2 nausea 16.4% vs
    # PS 0-1 8.5%, vomiting 11.9% vs 7.6% in mCRC pooled analysis)
    # [sargent2009]; Lyman et al. Cancer 2011 (neutropenia risk model, ECOG
    # not independently significant) [lyman2011]; Feliu et al. The Oncologist
    # 2020 (ECOG OR 1.30, p=0.236, not independently significant) [feliu2020];
    # Hurria et al. JCO 2011 (overall toxicity prediction) [hurria2011].
    # Per-SOC multipliers are clinical estimates; cited papers provide
    # directional support but not per-organ HRs by ECOG.
    "poor_ecog": {
        "Blood and lymphatic system disorders": 1.6,
        "Infections and infestations": 2.0,
        "Respiratory, thoracic and mediastinal disorders": 1.5,
        "General disorders and administration site conditions": 1.5,
        "Metabolism and nutrition disorders": 1.4,
        "Nervous system disorders": 1.3,
        "Gastrointestinal disorders": 1.4,
        "Cardiac disorders": 1.6,
        "Vascular disorders": 1.4,
        "Psychiatric disorders": 1.3,
    },
    "good_ecog": {
        "General disorders and administration site conditions": 0.85,
        "Infections and infestations": 0.80,
        "Respiratory, thoracic and mediastinal disorders": 0.85,
        "Cardiac disorders": 0.85,
    },
    # High comorbidity burden (CCI ≥2): higher AE risk.
    # Directional support: Gross et al. Cancer 2007 (heart failure OR 0.49 for
    # chemo receipt, i.e. high-comorbidity patients underrepresented; no
    # per-AE HRs) [gross2007]; Søgaard et al. Clin Epidemiol 2013 (review:
    # comorbidity worsens cancer survival, no per-SOC AE HRs) [sogaard2013].
    # Per-SOC multipliers are clinical estimates; cited papers provide
    # directional evidence only.
    "high_comorbidity": {
        "Blood and lymphatic system disorders": 1.4,
        "Infections and infestations": 1.5,
        "Cardiac disorders": 1.6,
        "Vascular disorders": 1.5,
        "Respiratory, thoracic and mediastinal disorders": 1.4,
        "General disorders and administration site conditions": 1.3,
        "Metabolism and nutrition disorders": 1.3,
        "Nervous system disorders": 1.2,
        "Renal and urinary disorders": 1.4,
        "Psychiatric disorders": 1.2,
        "Gastrointestinal disorders": 1.2,
    },
    "low_comorbidity": {
        "Cardiac disorders": 0.7,
        "Vascular disorders": 0.8,
        "Infections and infestations": 0.85,
    },
}


# Risk fields that have evidence-based defaults — only serialized when non-default.
_RISK_FIELD_DEFAULTS: dict[str, float] = {
    "os_multiplier_older": 0.88,
    "os_multiplier_younger": 1.12,
    "os_multiplier_poor_ecog": 0.62,
    "os_multiplier_high_comorbidity": 0.72,
    "death_prob_older_delta": 0.05,
    "death_prob_younger_delta": -0.05,
    "death_prob_poor_ecog_delta": 0.10,
    "death_prob_high_comorbidity_delta": 0.06,
    "fraction_high_comorbidity": 0.30,
    "male_weight_multiplier": 1.08,
    "female_weight_multiplier": 0.92,
    "male_height_multiplier": 1.04,
    "female_height_multiplier": 0.96,
    "patient_ae_propensity_sigma": 0.4,
}


# ============================================================================
# DATA CLASSES — Describe a clinical trial's characteristics
# ============================================================================

@dataclass
class AdverseEvent:
    """A single adverse event with its aggregate probability."""

    term: str
    organ_system: str
    probability: float  # fraction of patients affected (0–1)
    is_serious: bool = False

    def __post_init__(self):
        if self.probability < 0.0:
            raise ValueError(
                f"AE probability must be >= 0, got {self.probability} for '{self.term}'"
            )

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "organ_system": self.organ_system,
            "probability": self.probability,
            "is_serious": self.is_serious,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AdverseEvent":
        return cls(
            term=d["term"],
            organ_system=d["organ_system"],
            probability=d["probability"],
            is_serious=d.get("is_serious", False),
        )


@dataclass
class TrialConfig:
    """Complete configuration describing one clinical trial.

    All trial parameters — identity, demographics, survival, treatment
    cycles, adverse events, and risk modifiers — live in one flat dataclass.
    Only ``trial_id``, ``title``, ``condition``, ``treatment``,
    ``n_enrolled``, and ``adverse_events`` are required; everything else
    has evidence-based defaults.
    """

    # ── Trial identity (required) ──
    trial_id: str
    title: str
    condition: str
    treatment: str
    n_enrolled: int
    adverse_events: list[AdverseEvent]

    # ── Arm label (for multi-arm trials) ──
    arm_name: str = ""

    # ── Demographics ──
    age_mean: float = 64.0
    age_sd: float = 8.3
    age_min: int = 18
    age_max: int = 90
    fraction_male: float = 0.70
    ecog_distribution: dict[int, float] = field(
        default_factory=lambda: {0: 0.43, 1: 0.43, 2: 0.14}
    )
    weight_mean: float = 75.8
    weight_sd: float = 16.25
    height_mean: float = 169.9
    height_sd: float = 9.04
    race_distribution: dict[str, float] = field(
        default_factory=lambda: {"White": 0.972, "Black": 0.009, "Other": 0.019}
    )
    region_distribution: dict[str, float] = field(
        default_factory=lambda: {"United States": 1.0}
    )

    # ── Survival ──
    # If `os_km_csv` is set, the generator bootstraps per-patient (time, event)
    # directly from the matching `arm` rows in that CSV — this is treated as
    # ground truth for the arm's overall survival distribution, and the
    # parametric fields below are ignored. Otherwise, OS is sampled from a
    # Weibull parameterized by `median_os_months` / `os_weibull_shape`.
    os_km_csv: Optional[str] = None
    # When bootstrapping from `os_km_csv`, controls how strongly per-patient
    # risk (age / ECOG / comorbidity) correlates with the assigned survival
    # time. 0 = perfect rank correlation (highest-risk patient gets shortest
    # time); larger values inject Gaussian noise that breaks the correlation.
    # The marginal KM curve is preserved regardless. Default ≈ 0.5 Spearman.
    km_rank_noise_sigma: float = 0.5
    median_os_months: float = 12.0
    os_weibull_shape: float = 1.2
    death_probability: float = 0.80
    max_followup_months: float = 38.1

    # ── Treatment cycles ──
    cycle_length_days: int = 21
    n_induction_cycles: int = 4
    induction_ae_fraction: float = 0.85

    # ── Risk modifiers (evidence-based defaults) ──
    ae_risk_modifiers: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            k: dict(v) for k, v in _DEFAULT_AE_RISK_MODIFIERS.items()
        }
    )
    os_multiplier_older: float = 0.88
    os_multiplier_younger: float = 1.12
    os_multiplier_poor_ecog: float = 0.62
    os_multiplier_high_comorbidity: float = 0.72
    death_prob_older_delta: float = 0.05
    death_prob_younger_delta: float = -0.05
    death_prob_poor_ecog_delta: float = 0.10
    death_prob_high_comorbidity_delta: float = 0.06
    fraction_high_comorbidity: float = 0.30
    male_weight_multiplier: float = 1.08
    female_weight_multiplier: float = 0.92
    male_height_multiplier: float = 1.04
    female_height_multiplier: float = 0.96
    patient_ae_propensity_sigma: float = 0.4

    # ── Serialization ──

    # Directory the config was loaded from — used to resolve relative
    # `os_km_csv` paths. Not serialized.
    _source_dir: Optional[str] = field(default=None, repr=False, compare=False)

    # Fields that need special serialization or are handled separately
    _SKIP_IN_AUTO_SERIAL = {"adverse_events", "ae_risk_modifiers", "_source_dir"} | set(_RISK_FIELD_DEFAULTS)

    def to_dict(self) -> dict:
        d: dict = {}
        for f in fields(self):
            if f.name in self._SKIP_IN_AUTO_SERIAL:
                continue
            val = getattr(self, f.name)
            if f.name == "ecog_distribution":
                val = {str(k): v for k, v in val.items()}
            d[f.name] = val
        d["adverse_events"] = [ae.to_dict() for ae in self.adverse_events]
        # Risk fields: only serialize when non-default
        default_mods = {k: dict(v) for k, v in _DEFAULT_AE_RISK_MODIFIERS.items()}
        if self.ae_risk_modifiers != default_mods:
            d["ae_risk_modifiers"] = self.ae_risk_modifiers
        for fname, default_val in _RISK_FIELD_DEFAULTS.items():
            val = getattr(self, fname)
            if val != default_val:
                d[fname] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrialConfig":
        flat = dict(d)
        # Flatten nested sections (backward compat with old JSON format)
        for section in ("demographics", "survival", "risk_profile"):
            if section in flat and isinstance(flat[section], dict):
                nested = flat.pop(section)
                for k, v in nested.items():
                    if k not in flat:
                        flat[k] = v
        # Parse adverse_events
        if "adverse_events" in flat:
            flat["adverse_events"] = [
                AdverseEvent.from_dict(ae) for ae in flat["adverse_events"]
            ]
        # Convert ecog_distribution keys to int
        if "ecog_distribution" in flat:
            flat["ecog_distribution"] = {
                int(k): v for k, v in flat["ecog_distribution"].items()
            }
        # Deep-merge ae_risk_modifiers with defaults
        if "ae_risk_modifiers" in flat:
            merged = {k: dict(v) for k, v in _DEFAULT_AE_RISK_MODIFIERS.items()}
            for factor, organs in flat["ae_risk_modifiers"].items():
                if factor not in merged:
                    merged[factor] = {}
                merged[factor].update(organs)
            flat["ae_risk_modifiers"] = merged
        # Filter to valid fields
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in flat.items() if k in valid})

    def save_json(self, filepath: str) -> None:
        """Save this config to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "TrialConfig":
        """Load a TrialConfig from a JSON file."""
        with open(filepath) as f:
            cfg = cls.from_dict(json.load(f))
        cfg._source_dir = os.path.dirname(os.path.abspath(filepath))
        return cfg

    def resolve_path(self, path: str) -> str:
        """Resolve a (possibly relative) path against the config's source dir,
        falling back to cwd if the source-dir candidate does not exist."""
        if os.path.isabs(path):
            return path
        candidates = []
        if self._source_dir:
            candidates.append(os.path.normpath(os.path.join(self._source_dir, path)))
        candidates.append(os.path.normpath(os.path.join(os.getcwd(), path)))
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]

    def load_km_samples(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Load (time, event) arrays for this arm from `os_km_csv`.

        Returns None if no CSV is configured. Filters by `arm_name` when the
        CSV contains an `arm` column; otherwise uses all rows.
        """
        if not self.os_km_csv:
            return None
        path = self.resolve_path(self.os_km_csv)
        # Lightweight CSV read to avoid a hard pandas dependency in this module.
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError(f"KM CSV is empty: {path}")
        if "arm" in rows[0] and self.arm_name:
            rows = [r for r in rows if r["arm"] == self.arm_name]
            if not rows:
                raise ValueError(
                    f"No rows in {path} match arm_name={self.arm_name!r}"
                )
        times = np.asarray([float(r["time"]) for r in rows], dtype=float)
        events = np.asarray([int(float(r["event"])) for r in rows], dtype=int)
        return times, events

    def get_ae_by_organ(self) -> dict[str, list[AdverseEvent]]:
        """Group adverse events by organ system."""
        result: dict[str, list[AdverseEvent]] = {}
        for ae in self.adverse_events:
            result.setdefault(ae.organ_system, []).append(ae)
        return result

    def get_organ_systems(self) -> list[str]:
        """Return sorted list of unique organ systems."""
        return sorted({ae.organ_system for ae in self.adverse_events})

    # ── Subcategory generation ──

    def build_subcategories(self) -> list["PatientSubcategory"]:
        """Auto-generate patient archetypes from demographics.

        Crosses: age_bin × sex × ecog_bin × comorbidity_level
        to produce weighted archetypes with per-organ AE risk multipliers.
        """
        organ_systems = self.get_organ_systems()
        age_mid = int(self.age_mean)

        ecog_bins = [
            ("good_ecog" if v <= 1 else "poor_ecog", v, f)
            for v, f in sorted(self.ecog_distribution.items()) if f > 0
        ]
        age_bins = [
            ("younger", (self.age_min, age_mid - 1), 0.5),
            ("older", (age_mid, self.age_max), 0.5),
        ]
        sex_bins = [
            ("male", "M", self.fraction_male),
            ("female", "F", 1.0 - self.fraction_male),
        ]
        frac_high = self.fraction_high_comorbidity
        comorbidity_bins = [
            ("low_comorbidity", "low", 1.0 - frac_high),
            ("high_comorbidity", "high", frac_high),
        ]

        wt_mult = {"M": self.male_weight_multiplier, "F": self.female_weight_multiplier}
        ht_mult = {"M": self.male_height_multiplier, "F": self.female_height_multiplier}

        subcategories: list[PatientSubcategory] = []
        for (age_lbl, age_rng, age_w), (sex_lbl, sex_val, sex_w), \
            (ecog_lbl, ecog_val, ecog_w), (comorbid_lbl, comorbid_val, comorbid_w) \
                in itertools.product(age_bins, sex_bins, ecog_bins, comorbidity_bins):
            factor_labels = [age_lbl, sex_lbl, ecog_lbl, comorbid_lbl]
            subcategories.append(PatientSubcategory(
                name="_".join(factor_labels),
                age_range=age_rng,
                sex=sex_val,
                ecog=ecog_val,
                comorbidity=comorbid_val,
                weight_mean_kg=self.weight_mean * wt_mult[sex_val],
                weight_sd_kg=self.weight_sd,
                height_mean_cm=self.height_mean * ht_mult[sex_val],
                height_sd_cm=self.height_sd,
                os_multiplier=self._os_multiplier(age_lbl, ecog_lbl, comorbid_lbl),
                death_probability=self._death_prob(
                    self.death_probability, age_lbl, ecog_lbl, comorbid_lbl,
                ),
                ae_risk_multipliers=self._combine_multipliers(organ_systems, factor_labels),
                population_weight=age_w * sex_w * ecog_w * comorbid_w,
            ))

        total_w = sum(s.population_weight for s in subcategories)
        if total_w > 0:
            for s in subcategories:
                s.population_weight /= total_w

        return subcategories

    def _combine_multipliers(
        self,
        organ_systems: list[str],
        factor_labels: list[str],
    ) -> dict[str, float]:
        """Combine risk modifiers from multiple factors multiplicatively."""
        result: dict[str, float] = {}
        for organ in organ_systems:
            mult = 1.0
            normalized = _normalize_organ(organ)
            for label in factor_labels:
                modifiers = self.ae_risk_modifiers.get(label, {})
                m = modifiers.get(organ, modifiers.get(normalized, 1.0))
                mult *= m
            result[organ] = round(mult, 3)
        return result

    def _os_multiplier(self, age_label: str, ecog_label: str, comorbid_label: str) -> float:
        """Compute OS multiplier from patient factors."""
        os_mults = {
            "older": self.os_multiplier_older,
            "younger": self.os_multiplier_younger,
            "poor_ecog": self.os_multiplier_poor_ecog,
            "high_comorbidity": self.os_multiplier_high_comorbidity,
        }
        mult = 1.0
        for label in (age_label, ecog_label, comorbid_label):
            mult *= os_mults.get(label, 1.0)
        return mult

    def _death_prob(
        self, base: float, age_label: str, ecog_label: str, comorbid_label: str
    ) -> float:
        """Adjust death probability by patient factors."""
        death_deltas = {
            "older": self.death_prob_older_delta,
            "younger": self.death_prob_younger_delta,
            "poor_ecog": self.death_prob_poor_ecog_delta,
            "high_comorbidity": self.death_prob_high_comorbidity_delta,
        }
        p = base
        for label in (age_label, ecog_label, comorbid_label):
            p += death_deltas.get(label, 0.0)
        return round(np.clip(p, 0.10, 0.99), 3)


@dataclass
class PatientSubcategory:
    """A patient archetype with demographic profile + AE risk modifiers."""

    name: str
    age_range: tuple[int, int]
    sex: str  # "M" or "F"
    ecog: int
    comorbidity: str  # "low" or "high"

    # Demographic sampling parameters
    weight_mean_kg: float = 75.0
    weight_sd_kg: float = 15.0
    height_mean_cm: float = 170.0
    height_sd_cm: float = 8.0

    # Survival modifiers relative to trial-level
    os_multiplier: float = 1.0  # Multiplied with trial median OS
    death_probability: float = 0.80

    # Per-organ-system AE risk multipliers
    ae_risk_multipliers: dict[str, float] = field(default_factory=dict)

    # Population weight (probability of this subcategory)
    population_weight: float = 1.0


# ============================================================================
# SAMPLING ENGINE
# ============================================================================


def _pick(dist: dict[str, float], rng: np.random.Generator) -> str:
    """Sample a key from a {label: weight} distribution."""
    keys = list(dist.keys())
    w = np.array([dist[k] for k in keys], dtype=float)
    return rng.choice(keys, p=w / w.sum()) if w.sum() > 0 else keys[0]


class SyntheticPatientGenerator:
    """Generates synthetic patient populations from a TrialConfig.

    Usage:
        config = load_nct03041311()  # or build your own TrialConfig
        gen = SyntheticPatientGenerator(config)
        rows = gen.generate(n=200, seed=42)
        gen.save_csv(rows, "output.csv")
    """

    def __init__(
        self,
        config: TrialConfig,
        subcategories: Optional[list[PatientSubcategory]] = None,
    ):
        self.config = config

        # Build subcategories automatically, or use provided ones
        if subcategories is not None:
            self.subcategories = subcategories
        else:
            self.subcategories = config.build_subcategories()

        # Combined list of all (event_key, organ_system, base_prob)
        self._all_aes: list[tuple[str, str, float]] = []
        for ae in config.adverse_events:
            key = f"{'SAE' if ae.is_serious else 'AE'}:{ae.term}"
            self._all_aes.append((key, ae.organ_system, ae.probability))

        # Ground-truth KM bootstrap pool (None when no os_km_csv configured).
        self._km_samples: Optional[tuple[np.ndarray, np.ndarray]] = (
            config.load_km_samples()
        )

    def generate(
        self,
        n: int = 200,
        seed: int = 42,
    ) -> list[dict]:
        """Generate n synthetic patients with adverse event histories.

        Returns list of dicts: {patient_id, time, event, demographic_info}
        Each patient produces multiple rows (treatment_start + AEs + terminal).
        """
        rng = np.random.default_rng(seed)
        rows: list[dict] = []

        weights = np.array([s.population_weight for s in self.subcategories])
        weights = weights / weights.sum()
        cfg = self.config

        # Pre-sample subcategories for all patients so we can do
        # rank-correlated KM assignment in one shot below.
        subcat_idx = rng.choice(len(self.subcategories), size=n, p=weights)
        patient_subcats = [self.subcategories[k] for k in subcat_idx]

        # If a ground-truth KM CSV is wired up, pre-assign (os_time, is_death)
        # to every patient using a rank-correlation scheme:
        #   1. Bootstrap n rows from the empirical (time, event) pool.
        #   2. Sort those rows by time ascending.
        #   3. Score each patient by frailty (-log os_multiplier) + Gaussian
        #      noise scaled by `km_rank_noise_sigma`.
        #   4. Argsort patients by score descending, then zip: the
        #      highest-frailty patient is paired with the shortest bootstrapped
        #      time, the lowest-frailty patient with the longest, etc.
        # This preserves the marginal KM curve exactly (we only permute the
        # mapping from rows to patients) while inducing within-cohort
        # correlation between subcategory risk and survival.
        pre_os_time: Optional[np.ndarray] = None
        pre_is_death: Optional[np.ndarray] = None
        if self._km_samples is not None:
            km_times, km_events = self._km_samples
            boot_idx = rng.integers(0, len(km_times), size=n)
            boot_times = km_times[boot_idx]
            boot_events = km_events[boot_idx]
            time_order = np.argsort(boot_times, kind="stable")
            sorted_times = boot_times[time_order]
            sorted_events = boot_events[time_order]

            risk = np.array(
                [-np.log(max(s.os_multiplier, 1e-6)) for s in patient_subcats]
            )
            risk_noisy = risk + rng.normal(0.0, cfg.km_rank_noise_sigma, size=n)
            patient_order = np.argsort(-risk_noisy, kind="stable")

            pre_os_time = np.empty(n, dtype=float)
            pre_is_death = np.empty(n, dtype=bool)
            pre_os_time[patient_order] = sorted_times
            pre_is_death[patient_order] = sorted_events.astype(bool)

        for i in range(n):
            patient_id = f"SYN-{i + 1:04d}"

            # 1. Subcategory (pre-sampled above)
            subcat = patient_subcats[i]

            # 2. Generate demographics (static)
            demographics = self._sample_demographics(subcat, rng)

            # 3+4. Overall survival time and terminal event.
            if pre_os_time is not None:
                os_time = float(pre_os_time[i])
                is_death = bool(pre_is_death[i])
                terminal_event = "death" if is_death else "censored"
            else:
                adjusted_median = cfg.median_os_months * subcat.os_multiplier
                lam = adjusted_median / (np.log(2) ** (1.0 / cfg.os_weibull_shape))
                os_time = float(np.clip(lam * rng.weibull(cfg.os_weibull_shape), 0.5, cfg.max_followup_months))
                is_death = rng.random() < subcat.death_probability
                terminal_event = "death" if is_death else "censored"
                if not is_death:
                    os_time = float(np.clip(os_time * rng.uniform(0.6, 1.0), 0.5, cfg.max_followup_months))

            # 5. Sample AE events
            ae_events = self._sample_ae_events(subcat, os_time, rng)

            # 6. Emit rows
            demo_json = json.dumps(demographics)

            rows.append({
                "patient_id": patient_id,
                "time": 0.0,
                "event": "treatment_start",
                "demographic_info": demo_json,
            })

            for event_time, event_name in ae_events:
                rows.append({
                    "patient_id": patient_id,
                    "time": round(event_time, 2),
                    "event": event_name,
                    "demographic_info": demo_json,
                })

            rows.append({
                "patient_id": patient_id,
                "time": round(os_time, 2),
                "event": terminal_event,
                "demographic_info": demo_json,
            })

        return rows

    def _sample_demographics(
        self, subcat: PatientSubcategory, rng: np.random.Generator
    ) -> dict:
        """Generate static demographic info for a patient."""
        age = int(rng.integers(subcat.age_range[0], subcat.age_range[1] + 1))
        weight = float(np.clip(rng.normal(subcat.weight_mean_kg, subcat.weight_sd_kg), 40, 160))
        height = float(np.clip(rng.normal(subcat.height_mean_cm, subcat.height_sd_cm), 140, 200))
        return {
            "age": age,
            "sex": subcat.sex,
            "ecog": subcat.ecog,
            "weight_kg": round(weight, 1),
            "height_cm": round(height, 1),
            "bmi": round(weight / ((height / 100) ** 2), 1),
            "race": _pick(self.config.race_distribution, rng),
            "region": _pick(self.config.region_distribution, rng),
            "subcategory": subcat.name,
            "comorbidity": subcat.comorbidity,
        }

    def _sample_ae_events(
        self,
        subcat: PatientSubcategory,
        total_time_months: float,
        rng: np.random.Generator,
    ) -> list[tuple[float, str]]:
        """Sample adverse event occurrences over the patient's study duration.

        Uses induction-phase front-loading: the configured `induction_ae_fraction`
        controls what fraction of each AE's total probability budget is spent
        during the induction cycles vs. maintenance cycles.
        """
        events: list[tuple[float, str]] = []
        cycle_length_months = self.config.cycle_length_days / 30.44
        n_total_cycles = max(1, int(total_time_months / cycle_length_months))
        n_induction = min(self.config.n_induction_cycles, n_total_cycles)
        n_maintenance = n_total_cycles - n_induction

        induction_frac = self.config.induction_ae_fraction
        maintenance_frac = 1.0 - induction_frac

        # Patient-level AE propensity (lognormal inter-patient variance)
        sigma = self.config.patient_ae_propensity_sigma
        patient_ae_mult = float(rng.lognormal(0, sigma))

        for cycle_idx in range(n_total_cycles):
            cycle_start = cycle_idx * cycle_length_months
            if cycle_start >= total_time_months:
                break

            is_induction = cycle_idx < n_induction

            for event_key, organ_system, base_p in self._all_aes:
                mults = subcat.ae_risk_multipliers
                multiplier = mults.get(organ_system, mults.get(_normalize_organ(organ_system), 1.0))
                adjusted_p = max(0.0, base_p * multiplier * patient_ae_mult)

                if is_induction and n_induction > 0:
                    per_cycle_p = adjusted_p * induction_frac / n_induction
                elif not is_induction and n_maintenance > 0:
                    per_cycle_p = adjusted_p * maintenance_frac / n_maintenance
                else:
                    per_cycle_p = adjusted_p / n_total_cycles

                n_occurrences = int(rng.poisson(per_cycle_p))
                for _ in range(n_occurrences):
                    # Cluster onset near start of cycle (early visits)
                    time = cycle_idx * cycle_length_months + (rng.random() * cycle_length_months / 3)
                    event_time = min(time, total_time_months)
                    event_name = event_key.split(":", 1)[1]
                    events.append((round(event_time, 2), event_name))

        events.sort(key=lambda x: x[0])
        return events

    # ── I/O helpers ─────────────────────────────────────────────────────

    @staticmethod
    def save_csv(rows: list[dict], filepath: str) -> None:
        """Save generated patient data to CSV."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["patient_id", "time", "event", "demographic_info"]
            )
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def print_summary(rows: list[dict]) -> None:
        """Print summary statistics of the generated population."""
        patients: dict[str, list] = {}
        for row in rows:
            patients.setdefault(row["patient_id"], []).append(row)

        n_patients = len(patients)
        if n_patients == 0:
            print("No patients generated.")
            return

        non_meta = {"treatment_start", "death", "censored"}
        n_deaths = sum(
            1 for events in patients.values()
            if any(e["event"] == "death" for e in events)
        )
        n_censored = n_patients - n_deaths

        os_times = np.array([
            events[-1]["time"] for events in patients.values()
            if events[-1]["event"] in ("death", "censored")
        ])
        median_os = float(np.median(os_times)) if len(os_times) > 0 else 0

        ae_counts: Counter[str] = Counter()
        for events in patients.values():
            for e in events:
                if e["event"] not in non_meta:
                    ae_counts[e["event"]] += 1
        total_ae_count = sum(ae_counts.values())

        subcat_counts = Counter(
            json.loads(events[0]["demographic_info"]).get("subcategory", "unknown")
            for events in patients.values()
        )

        print("=" * 70)
        print("SYNTHETIC PATIENT POPULATION SUMMARY")
        print("=" * 70)
        print(f"  Total patients:     {n_patients}")
        print(f"  Deaths:             {n_deaths} ({100 * n_deaths / n_patients:.1f}%)")
        print(f"  Censored:           {n_censored} ({100 * n_censored / n_patients:.1f}%)")
        print(f"  Median OS:          {median_os:.1f} months")
        print(f"  Unique AE types:    {len(ae_counts)}")
        print(f"  Total AE events:    {total_ae_count}")
        print(f"  Mean AEs/patient:   {total_ae_count / n_patients:.1f}")
        print()
        print("  Subcategory distribution:")
        for sc, count in subcat_counts.most_common():
            print(f"    {sc}: {count} ({100 * count / n_patients:.1f}%)")
        print()
        print("  Top 15 adverse events:")
        for ae, count in ae_counts.most_common(15):
            pct = 100 * count / total_ae_count if total_ae_count > 0 else 0
            print(f"    {ae}: {count} ({pct:.1f}%)")
        print("=" * 70)


# ============================================================================
# CLI
# ============================================================================
def generate_combined(
    configs: list[TrialConfig],
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic patients from multiple configs and combine into one dataset.

    Each config represents a trial arm and emits exactly ``config.n_enrolled``
    synthetic patients. Patients are tagged with an ``arm`` field derived from
    the config's ``arm_name`` attribute (falls back to ``treatment``). Patient
    IDs are globally unique across arms.
    """
    all_rows: list[dict] = []
    patient_offset = 0

    for config in configs:
        arm_label = getattr(config, "arm_name", None) or config.treatment
        arm_n = config.n_enrolled
        gen = SyntheticPatientGenerator(config)
        rows = gen.generate(n=arm_n, seed=seed)

        # Re-number patient IDs to be globally unique and add arm column
        for row in rows:
            old_num = int(row["patient_id"].split("-")[1])
            row["patient_id"] = f"SYN-{old_num + patient_offset:04d}"
            row["arm"] = arm_label

        patient_offset += arm_n
        all_rows.extend(rows)

    return all_rows


def save_csv_with_arm(rows: list[dict], filepath: str) -> None:
    """Save generated patient data (with arm column) to CSV."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["patient_id", "arm", "time", "event", "demographic_info"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic patient adverse event data from clinical trial results. "
                    "Each config emits exactly its n_enrolled patients."
    )
    parser.add_argument(
        "config", type=str, nargs="+",
        help="Path(s) to trial config JSON file(s). Multiple configs produce a combined multi-arm CSV.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output CSV path (default: datasets/<trial_id>/synthetic_patients.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    configs = [TrialConfig.load_json(p) for p in args.config]

    if len(configs) == 1:
        config = configs[0]
        output_path = args.output or f"datasets/{config.trial_id}/synthetic_patients.csv"

        print(f"Trial: {config.trial_id} — {config.title}")
        print(f"Generating {config.n_enrolled} synthetic patients (seed={args.seed})...")

        gen = SyntheticPatientGenerator(config)
        rows = gen.generate(n=config.n_enrolled, seed=args.seed)

        gen.print_summary(rows)
        gen.save_csv(rows, output_path)
        print(f"\nData saved to: {output_path}")
        print(f"Total rows: {len(rows)}")
    else:
        trial_id = configs[0].trial_id
        output_path = args.output or f"datasets/{trial_id}/synthetic_patients.csv"

        print(f"Trial: {trial_id} — {len(configs)} arms")
        for i, cfg in enumerate(configs):
            arm_label = getattr(cfg, "arm_name", None) or cfg.treatment
            print(f"  Arm {i + 1}: {arm_label} (n_enrolled={cfg.n_enrolled})")
        print(f"Generating per-arm n_enrolled patients (seed={args.seed})...")

        rows = generate_combined(configs, seed=args.seed)

        save_csv_with_arm(rows, output_path)
        total_patients = sum(c.n_enrolled for c in configs)
        print(f"\nCombined data saved to: {output_path}")
        print(f"Total rows: {len(rows)}")
        print(f"Total patients: {total_patients}")


if __name__ == "__main__":
    main()
