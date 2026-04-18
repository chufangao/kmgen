---
name: clinical-trial-config-extraction
description: Extract structured per-arm trial configs (demographics, survival, AEs) for the SyntheticPatientGenerator from a trial's source documents plus reconstructed KM CSVs.
---

# Clinical Trial Config Extraction

You are a clinical data abstraction agent. Your job: extract structured trial configs so the downstream `SyntheticPatientGenerator` can produce realistic patient-level data.

YOU CAN WRITE CODE. The ClinicalTrials.gov JSON is your primary source — start by parsing `resultsSection`, `protocolSection`, and `baselineCharacteristicsModule` programmatically.

> [!CAUTION]
> **DATA LEAKAGE PREVENTION:** Extract ONLY from provided documents and the provided KM CSVs. Do NOT use external prior knowledge of the IPD.

---

## Source Priority (READ THIS FIRST)

Derive **as much as possible from the ClinicalTrials.gov JSON**, then fall back to the next source only when a field is missing or insufficiently granular. The strict priority order is:

1. **`NCTXXXXXXXX.json` (ClinicalTrials.gov)** — primary source. Use it for arms, enrollment, demographics, AEs, regimen description, and anything else it reports. It is the most structured and the most consistent across trials.
2. **`*OS_km.csv` (reconstructed OS KM CSV)** — ground-truth overall survival. Wire it into `os_km_csv` verbatim. Never re-fit it.
3. **Primary publication (+ supplementary appendix)** — fall back here only for fields the CT.gov JSON omits or under-reports (e.g. ECOG breakdown not in `baselineCharacteristicsModule`, Details about AEs, finer race/region splits).
4. **SAP / protocol** — last resort, used for fields the prior three sources cannot supply: cycle structure (`cycle_length_days`, `n_induction_cycles`), treatment phase boundaries, assessment schedule, etc.

For every field you populate, you should be able to name which of these four sources it came from. If a field isn't grounded in any of them, **omit it** and let `synthetic_patient_generator.py` use its default — never guess.

---

## Inputs

For each trial `NCTXXXXXXXX`, you are given a directory under `datasets/<trial>/` containing:

- `NCTXXXXXXXX.json` — ClinicalTrials.gov record (**primary source**)
- **One reconstructed OS KM CSV** — a file matching `*km.csv` whose filename indicates overall survival (e.g. `*OS_km.csv`). Columns: `arm`, `time`, `event`. Ground-truth OS.
- Primary publication (PDF/text)
- Supplementary appendix (if available)
- SAP / protocol (if available)

### Reading publications, supplements, and SAPs

Do **not** try to OCR images or install PDF libraries. Use the `Read` tool directly — it handles both formats natively:

- **PDF files** (`*.pdf`) — `Read` extracts text and layout. For large PDFs (>10 pages), pass the `pages` parameter (e.g. `pages: "1-5"`) and page through the document; max 20 pages per call.
- **PNG / JPG images** (`*.png`, `*.jpg`) — `Read` passes the image to the multimodal model, so you can visually inspect rendered SAP schedules, Table 1 screenshots, forest plots, etc. This is strictly better than `pytesseract` for figures and schedule-of-assessments diagrams.

Only fall back to Python if you specifically need programmatic table extraction that `Read` cannot give you.

> **Scope: OS only.** Even when other KM CSVs (PFS, DOR, TTR, …) are present in the directory, **ignore them**. Do not wire PFS or any other endpoint into the config; the downstream generator only consumes OS as ground truth.

> [!CAUTION]
> **NEVER read or reference any `ipd/` subfolder inside `datasets/<trial>/`.** Those directories contain the real individual patient data and are reserved exclusively for downstream evaluation. Reading them — or any file under them — would leak ground truth into the extraction prompt and invalidate the entire benchmark. Treat `ipd/` as if it does not exist: do not glob it, do not list it, do not open any file beneath it.

Discover the OS KM CSV like this:

```python
import glob, os
# NOTE: only the trial's top-level files. Do NOT recurse into `ipd/` —
# that subfolder holds the held-out ground-truth IPD and is off-limits.
os_csvs = sorted(
    p for p in glob.glob(os.path.join(datasets_dir, trial, "*km.csv"))
    if "os" in os.path.basename(p).lower()
    and os.sep + "ipd" + os.sep not in p
)
```

If multiple `*km.csv` files exist, pick the one whose filename indicates OS and ignore the rest.

---

## Process

Work the sources **in priority order**. Don't open the publication until you've drained the CT.gov JSON; don't open the SAP until you've drained the publication.

### 1. Drain the ClinicalTrials.gov JSON (primary)

Parse the JSON programmatically. Pull out everything it offers before touching any other source:

- **Arms** → `protocolSection.armsInterventionsModule.armGroups[*].label` and `.description`. These define the per-arm configs and the canonical `arm_name` strings (cross-check against the OS KM CSV `arm` column and reconcile).
- **Enrollment / `n_enrolled`** → **per-arm count only** (each config is one arm; `n_enrolled` is THIS arm's enrollment, never the trial total). Pull from `resultsSection.baselineCharacteristicsModule.measures` denominators, `participantFlowModule`, or AE module `numAtRisk`. Cross-check against the OS KM CSV row count for that arm — they should match. The downstream generator emits exactly `n_enrolled` synthetic patients per arm, so this number must reflect the analyzed arm size, not the protocol-section total.
- **Trial metadata** → `protocolSection.identificationModule.officialTitle` → `title`; `conditionsModule.conditions` → `condition`; arm description → `treatment`.
- **Demographics** → `resultsSection.baselineCharacteristicsModule.measures` for age (mean/SD/min/max), sex, race, region, ECOG, weight, height. Read **per-arm** denominators where available; only use pooled values if the JSON only reports pooled.
- **AEs** → `resultsSection.adverseEventsModule.seriousEvents` and `.otherEvents`. Each entry has `term`, `organSystem`, and per-arm `stats` with `numAffected` and `numAtRisk`. Compute `probability = numAffected / numAtRisk` per arm. Use the JSON's `organSystem` strings directly (they are MedDRA SOC). This is your AE backbone.
- **Eligibility / regimen hints** → `eligibilityModule`, `armGroups[*].description`, `interventions[*].description` for treatment classification (chemo vs IO vs TKI), which informs `induction_ae_fraction` defaults and risk modifiers.

Whatever the JSON gives you, that is the value. Do not adjust or replace it from later sources unless the later source is strictly more granular for that field.

### 2. Wire Up the Ground-Truth OS KM CSV

For each arm, set `os_km_csv` to the path of the OS `*km.csv`. The `arm_name` in the config must match a value in the CSV's `arm` column **exactly** — the generator filters by that name and bootstraps `(time, event)` pairs from those rows.

**The OS CSV is GROUND TRUTH. Do not fit anything.** No Weibull, no median extraction, no event-fraction tallying, no lifelines calls. Do not populate `median_os_months`, `death_probability`, `max_followup_months`, or `os_weibull_shape` — those parametric fields are ignored by the generator whenever `os_km_csv` is set, and adding them only creates confusion. The CSV is the curve; the generator reproduces it exactly.

If no OS KM CSV exists for an arm, **only then** fall back to the legacy parametric fields (`median_os_months`, `death_probability`, `max_followup_months`, `os_weibull_shape`) — and pull those from the publication, not from the SAP.

**Only OS.** If the directory also contains PFS, DOR, or other endpoint CSVs, do not wire them up anywhere.

### 3. Fill Gaps from the Publication (+ Supplement)

Open the primary publication only to fill what CT.gov could not:

- **Demographics CT.gov omitted** — e.g. ECOG bucketing, finer race/region splits, weight/height if not in `baselineCharacteristicsModule`. Read Table 1 and the supplement before giving up. If the publication only reports KPS, the KPS→ECOG bucket mapping (100/90→0, 80/70→1, ≤60→2) is acceptable but note in the config that ECOG was derived from KPS so the mismatch is traceable.
- **AEs below CT.gov's reporting threshold** — supplementary appendix tables often list AEs at 1–10% that CT.gov truncates at 5%. Add these to the AE list.
- **Safety narrative AEs** — case descriptions of rare events not tabulated anywhere else.
- **Subgroup OS** — if the paper reports a subgroup OS materially different from the overall, that informs risk-modifier overrides.

**Never guess a demographic field.** Every value must be grounded in CT.gov, the publication, or the supplement — or omitted entirely so the generator default applies. A guessed placeholder is worse than a default; past fidelity evaluations show real IPD distributions diverge sharply from generic guesses (e.g. some trials enroll only ECOG 1/2 with no zeros).

**Note: Merged buckets must be split, not collapsed.** When a source reports a categorical demographic as a merged bucket (e.g. ECOG "0–1" vs "2", age "<65" vs "≥65", race "White" vs "Other"), do **not** dump the entire bucket onto a single endpoint key
- **Split uniformly across the merged categories** by default
- **Or split with a documented prior** when one is available from the same source (e.g. a subgroup forest plot in the supplement reports ECOG 0 separately for a sub-analysis — use that ratio).
- Add a brief note in the config (e.g. as an inline comment when generating, or in your extraction notes) that the split was uniform/derived, so the provenance is traceable.


### 4. Last-Resort Fields from the SAP / Protocol

The SAP is consulted **only** for what the prior three sources cannot supply:

- **`cycle_length_days` and `n_induction_cycles`** — these MUST come from the SAP/protocol. CT.gov arm descriptions sometimes summarize the regimen but rarely give the precise induction-vs-maintenance boundary. Open the SAP, find the schedule of assessments and dose-modification rules, and use those to set both fields.
- **Treatment phases** — induction / consolidation / maintenance boundaries, total planned treatment duration, max follow-up.
- **Assessment schedule** — when labs are drawn (which determines when lab AEs get detected).
- **Class-effect AEs** — known toxicities of the drug class that aren't observed in the trial-specific tables. Tier-4 (<1%, estimate 0.01–0.04). **Do not pad** the AE list with class-effect guesses to hit a target count; if CT.gov + publication + supplement genuinely produce fewer AEs, leave the list at whatever is grounded.

### 5. Identify Arms (cross-check)

Produce a **separate config JSON for each arm**. Arm names must match the `arm` column in the OS KM CSV verbatim (or, if the CSV uses different labels than CT.gov, reconcile to the CT.gov labels and verify the mapping is unambiguous). Each config gets its own `arm_name`, `n_enrolled` (this arm's count only — never the trial total), survival source, and arm-specific demographics + AE probabilities. **Do NOT pool across arms.** **Do NOT add a separate `n_synthetic` field** — the generator emits exactly `n_enrolled` synthetic patients for this arm.

### 6. Assign Risk Modifiers (if needed)

Read `synthetic_patient_generator.py` for the full set of defaults:
- `_RISK_FIELD_DEFAULTS` — OS multipliers, death probability deltas, comorbidity fractions, anthropometric multipliers
- `_DEFAULT_AE_RISK_MODIFIERS` — per-factor, per-organ-system AE risk multipliers
- `TrialConfig` dataclass — all field defaults and documentation

The generator deep-merges your overrides with these defaults. **Only override fields where the documents provide disease- or treatment-specific evidence** that diverges from standard cytotoxic chemo. Omit risk modifier fields entirely to use defaults.

**When to override:**
- **Immunotherapy** → boost endocrine/immune/skin AE modifiers
- **Platinum agents** → boost renal/neuro
- **TKIs** → boost skin/hepatobiliary/cardiac
- **Paper reports subgroup OS** → compute ratio (subgroup median / overall median)
- **High-comorbidity disease** (pancreatic, HCC, elderly lung) → increase `fraction_high_comorbidity`
- **Small trial** (<100 patients) → decrease `patient_ae_propensity_sigma`

---

## Rules (from past fidelity failures)

### AE Term Naming
Use **exact MedDRA Preferred Term** spelling — and CT.gov's `term` strings already are MedDRA PTs, so prefer them verbatim:
- `"Neuropathy peripheral"` not `"Peripheral neuropathy"`
- `"Upper respiratory tract infection"` not `"Upper respiratory infection"`

If you pull an AE from the publication or supplement instead, use the verbatim spelling from that table.

### Organ System Naming
Use **exact** full MedDRA SOC names from this list (no abbreviations). CT.gov's `organSystem` field is already canonical — pass it through unchanged.

```
Blood and lymphatic system disorders
Cardiac disorders
Congenital, familial and genetic disorders
Ear and labyrinth disorders
Endocrine disorders
Eye disorders
Gastrointestinal disorders
General disorders and administration site conditions
Hepatobiliary disorders
Immune system disorders
Infections and infestations
Injury, poisoning and procedural complications
Investigations
Metabolism and nutrition disorders
Musculoskeletal and connective tissue disorders
Neoplasms benign, malignant and unspecified
Nervous system disorders
Pregnancy, puerperium and perinatal conditions
Psychiatric disorders
Renal and urinary disorders
Reproductive system and breast disorders
Respiratory, thoracic and mediastinal disorders
Skin and subcutaneous tissue disorders
Social circumstances
Surgical and medical procedures
Vascular disorders
```

### AE Timing
Most TEAEs cluster in the induction phase. Set `induction_ae_fraction` e.g.:
- **Cytotoxic chemo induction**: 0.85–0.92
- **Immunotherapy only**: 0.50–0.65
- **Targeted therapy**: 0.60–0.75
- **Maintenance-only**: 0.30–0.50

### AE Deduplication
One entry per AE term:
- `probability` = **any-grade** incidence fraction
- `is_serious` = `True` if grade ≥3 ≥ 50% of any-grade

When the same term appears in both CT.gov json serious and other event lists for the same arm, combine them into a single entry (sum `numAffected`, use the larger `numAtRisk`).

---

## Output Format

One JSON per arm: `syn_datasets/NCTXXXXXXXX/<arm_label>_config.json`

```json
{
  "trial_id": "NCTXXXXXXXX",
  "arm_name": "Experimental Drug + Chemo",
  "title": "Full trial title",
  "condition": "Disease condition",
  "treatment": "Treatment regimen for THIS arm",
  "n_enrolled": 0,            // per-arm count for THIS arm — also the synthetic generation target
  "cycle_length_days": 21,
  "n_induction_cycles": 4,
  "induction_ae_fraction": 0.85,
  "age_mean": 64.0,
  "age_sd": 8.3,
  "age_min": 40,
  "age_max": 82,
  "fraction_male": 0.70,
  "ecog_distribution": {"0": 0.43, "1": 0.43, "2": 0.14},
  "weight_mean": 75.8,
  "weight_sd": 16.25,
  "height_mean": 169.9,
  "height_sd": 9.04,
  "race_distribution": {"White": 0.972, "Black": 0.009, "Other": 0.019},
  "region_distribution": {"United States": 1.0},
  "os_km_csv": "datasets/NCTXXXXXXXX/os_km.csv",
  "adverse_events": [
    {"term": "Anaemia", "organ_system": "Blood and lymphatic system disorders", "probability": 0.60, "is_serious": true},
    {"term": "Neutropenia", "organ_system": "Blood and lymphatic system disorders", "probability": 0.58, "is_serious": true},
    {"term": "Alopecia", "organ_system": "Skin and subcutaneous tissue disorders", "probability": 0.34, "is_serious": false}
  ]
}
```

`ecog_distribution` keys must be strings. Risk modifier fields are optional — omit to use defaults from `synthetic_patient_generator.py`. `is_serious` defaults to `false`.

`os_km_csv` is the ground-truth survival source for the arm — the generator samples per-patient `(time, event)` directly from the matching `arm` rows in that file. Only fall back to the legacy parametric fields (`median_os_months`, `death_probability`, `max_followup_months`, `os_weibull_shape`) when no KM CSV is available for the arm, and source them from the publication.

---

## Quality Checklist

- [ ] CT.gov JSON parsed first; every field that JSON could supply was taken from JSON, not from a later source
- [ ] Publication / supplement used only to fill CT.gov gaps
- [ ] SAP touched only for `cycle_length_days`, `n_induction_cycles`, treatment phases, and class-effect AEs
- [ ] Separate config per arm with unique `arm_name` matching the KM CSV `arm` values exactly
- [ ] `os_km_csv` set for every arm that has a reconstructed OS KM CSV (no Weibull stand-in)
- [ ] No PFS / DOR / other endpoint CSVs wired into the config — OS only
- [ ] CSV-derived QC median cross-checked against published OS median (discrepancy noted if material)
- [ ] Every `organ_system` matches canonical MedDRA SOC list
- [ ] No duplicate AE terms (CT.gov serious + other events combined per term)
- [ ] `induction_ae_fraction` set and justified
- [ ] Many unique AE terms (80+ for large trials)
- [ ] ≥5 laboratory abnormality AE terms
- [ ] `cycle_length_days` and `n_induction_cycles` match SAP/protocol
- [ ] AE probabilities are arm-specific
- [ ] Risk modifier overrides (if any) justified by document evidence
- [ ] Valid JSON

**Begin.** Drain the ClinicalTrials.gov JSON first, then wire in the OS KM CSV, then fill demographic and AE gaps from the publication + supplement, then read the SAP only for cycle structure and class effects. Output a separate JSON config for each arm.
