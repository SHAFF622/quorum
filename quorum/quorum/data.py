"""
Load public breast cancer cohorts from cBioPortal into one tidy table each.

Two cohorts:
- METABRIC (`brca_metabric`): 2,509 tumors, long follow-up. Main dataset.
- TCGA breast, PanCancer Atlas (`brca_tcga_pan_can_atlas_2018`): a separate set
  of patients with race and genetic ancestry. Used to replicate findings.

Every column is renamed to one shared vocabulary, so the same analysis code
runs on either cohort. Raw API responses are cached on disk, and the tidy
table gets a short "data card" (counts, missing values, a content hash) so
every claim can record exactly which data it was computed from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

API = "https://www.cbioportal.org/api"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"

ESR1 = 2099
TP53 = 7157

STUDIES = {
    "metabric": {
        "study": "brca_metabric",
        "mrna": "brca_metabric_mrna_median_all_sample_Zscores",
        "mutations": "brca_metabric_mutations",
    },
    "tcga": {
        "study": "brca_tcga_pan_can_atlas_2018",
        "mrna": "brca_tcga_pan_can_atlas_2018_rna_seq_v2_mrna_median_all_sample_Zscores",
        "mutations": "brca_tcga_pan_can_atlas_2018_mutations",
    },
}

# The shared vocabulary. Every analysis, claim and referee uses these names.
COLUMNS = {
    "os_months": "overall survival time in months",
    "os_event": "1 if the patient died (any cause), 0 if alive at last follow-up",
    "dss_months": "disease-specific survival time in months",
    "dss_event": "1 if the patient died of breast cancer; 0 if alive or died of another cause",
    "age": "age at diagnosis in years",
    "er_status": "estrogen receptor status: Positive or Negative",
    "her2_status": "HER2 status: Positive or Negative",
    "pam50": "molecular subtype: LumA, LumB, Her2, Basal, Normal (METABRIC also claudin-low)",
    "grade": "tumor grade 1 to 3",
    "tumor_size": "tumor size in mm",
    "nodes_positive": "number of positive lymph nodes",
    "stage": "tumor stage",
    "chemotherapy": "received chemotherapy: YES or NO (METABRIC only)",
    "hormone_therapy": "received hormone therapy: YES or NO (METABRIC only)",
    "histology": "histological subtype",
    "menopausal_state": "Pre or Post (METABRIC only)",
    "race": "self-reported race (TCGA only)",
    "ancestry": "genetic ancestry label, e.g. EUR, AFR (TCGA only)",
    "esr1_z": "ESR1 gene expression z-score: a continuous estrogen receptor score",
    "tp53_mutated": "1 if a TP53 mutation was found, 0 if sequenced and none found",
}

# cBioPortal attribute -> shared name, per cohort.
_RENAMES = {
    "metabric": {
        "OS_MONTHS": "os_months", "OS_STATUS": "os_status", "AGE_AT_DIAGNOSIS": "age",
        "ER_STATUS": "er_status", "HER2_STATUS": "her2_status", "CLAUDIN_SUBTYPE": "pam50",
        "GRADE": "grade", "TUMOR_SIZE": "tumor_size",
        "LYMPH_NODES_EXAMINED_POSITIVE": "nodes_positive", "TUMOR_STAGE": "stage",
        "CHEMOTHERAPY": "chemotherapy", "HORMONE_THERAPY": "hormone_therapy",
        "HISTOLOGICAL_SUBTYPE": "histology", "INFERRED_MENOPAUSAL_STATE": "menopausal_state",
        "VITAL_STATUS": "vital_status",
    },
    "tcga": {
        "OS_MONTHS": "os_months", "OS_STATUS": "os_status", "AGE": "age",
        "SUBTYPE": "pam50", "GRADE": "grade", "AJCC_PATHOLOGIC_TUMOR_STAGE": "stage",
        "ICD_O_3_HISTOLOGY": "histology", "RACE": "race",
        "GENETIC_ANCESTRY_LABEL": "ancestry", "DSS_MONTHS": "dss_months", "DSS_STATUS": "dss_status",
    },
}

NUMERIC = ["os_months", "dss_months", "age", "grade", "tumor_size", "nodes_positive", "esr1_z"]


@dataclass
class DataCard:
    """A short, checkable summary of a loaded cohort."""

    cohort: str
    patients: int
    with_survival: int
    deaths: int
    median_follow_up_months: float
    missing: dict[str, int]
    content_sha256: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Fetching (cached)
# ---------------------------------------------------------------------------

def _cached(path: Path, fetch) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def _get(url: str, **params) -> Any:
    r = httpx.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def _post(url: str, body: dict) -> Any:
    r = httpx.post(url, json=body, params={"projection": "SUMMARY"}, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_raw(cohort: str, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    """Download (or read from cache) the clinical, ESR1 and TP53 data for a cohort."""
    cfg = STUDIES[cohort]
    study = cfg["study"]
    base = cache_dir / cohort
    out = {}
    for kind in ("PATIENT", "SAMPLE"):
        out[kind.lower()] = _cached(
            base / f"clinical_{kind.lower()}.json",
            lambda kind=kind: _get(
                f"{API}/studies/{study}/clinical-data",
                clinicalDataType=kind, projection="SUMMARY",
            ),
        )
    out["esr1"] = _cached(
        base / "esr1.json",
        lambda: _post(
            f"{API}/molecular-profiles/{cfg['mrna']}/molecular-data/fetch",
            {"entrezGeneIds": [ESR1], "sampleListId": f"{study}_all"},
        ),
    )
    out["tp53"] = _cached(
        base / "tp53.json",
        lambda: _post(
            f"{API}/molecular-profiles/{cfg['mutations']}/mutations/fetch",
            {"entrezGeneIds": [TP53], "sampleListId": f"{study}_sequenced"},
        ),
    )
    out["sequenced"] = _cached(
        base / "sequenced.json",
        lambda: _get(f"{API}/sample-lists/{study}_sequenced/sample-ids"),
    )
    return out


# ---------------------------------------------------------------------------
# Tidying
# ---------------------------------------------------------------------------

def _pivot(rows: list[dict]) -> pd.DataFrame:
    """Long cBioPortal rows (one per attribute) -> one row per patient."""
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)[["patientId", "clinicalAttributeId", "value"]]
    wide = long.pivot_table(
        index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    return wide


def tidy(cohort: str, raw: dict[str, Any]) -> pd.DataFrame:
    """Turn raw API data into the shared-vocabulary table, one row per patient."""
    patient = _pivot(raw["patient"])
    sample = _pivot(raw["sample"])
    # Patient-level values win; sample-level fills attributes stored per sample.
    wide = patient.combine_first(sample) if not sample.empty else patient

    df = wide.rename(columns=_RENAMES[cohort])
    keep = [c for c in _RENAMES[cohort].values() if c in df.columns]
    df = df[keep].copy()
    df.index.name = "patient_id"

    # Survival: "1:DECEASED" -> 1, "0:LIVING" -> 0.
    status = df.pop("os_status")
    df["os_event"] = status.map(
        lambda s: 1.0 if str(s).startswith("1") else (0.0 if str(s).startswith("0") else np.nan)
    )

    # Disease-specific survival: only deaths from breast cancer count. A death
    # from another cause ends follow-up without counting as an event.
    if cohort == "metabric":
        vital = df.pop("vital_status")
        df["dss_months"] = df["os_months"]
        df["dss_event"] = vital.map({"Died of Disease": 1.0, "Died of Other Causes": 0.0, "Living": 0.0})
    else:
        dss = df.pop("dss_status")
        df["dss_event"] = dss.map(
            lambda s: 1.0 if str(s).startswith("1") else (0.0 if str(s).startswith("0") else np.nan)
        )

    if cohort == "tcga":
        df["pam50"] = df["pam50"].astype("string").str.replace("BRCA_", "", regex=False)
        df["er_status"] = pd.NA  # TCGA PanCancer clinical table has no ER IHC field
        df["her2_status"] = pd.NA

    # ESR1 expression (sample-level) -> patient. One tumor sample per patient here.
    esr1 = {r["patientId"]: r["value"] for r in raw["esr1"]}
    df["esr1_z"] = df.index.map(esr1)

    # TP53: mutated if any mutation; 0 only if the patient was actually sequenced.
    mutated = {r["patientId"] for r in raw["tp53"]}
    sequenced_samples = set(raw["sequenced"])
    sample_to_patient = {r["sampleId"]: r["patientId"] for r in raw["sample"]}
    sequenced = {sample_to_patient[s] for s in sequenced_samples if s in sample_to_patient}
    df["tp53_mutated"] = [
        1.0 if p in mutated else (0.0 if p in sequenced else np.nan) for p in df.index
    ]

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "er_status" in df.columns and cohort == "metabric":
        df["er_status"] = df["er_status"].replace({"Positve": "Positive"})

    df.insert(0, "cohort", cohort)
    return df.sort_index()


def data_card(df: pd.DataFrame) -> DataCard:
    """Counts and checks that every run records next to its claims."""
    surv = df.dropna(subset=["os_months", "os_event"])
    warnings = []
    if len(surv) < 0.5 * len(df):
        warnings.append("Fewer than half the patients have survival data.")
    if surv["os_event"].sum() < 100:
        warnings.append(
            f"Only {int(surv['os_event'].sum())} deaths: survival claims will be underpowered."
        )
    csv = df.to_csv(index=True).encode()
    return DataCard(
        cohort=str(df["cohort"].iloc[0]) if len(df) else "",
        patients=len(df),
        with_survival=len(surv),
        deaths=int(surv["os_event"].sum()),
        median_follow_up_months=float(surv["os_months"].median()) if len(surv) else 0.0,
        missing={c: int(df[c].isna().sum()) for c in df.columns if c != "cohort"},
        content_sha256=hashlib.sha256(csv).hexdigest(),
        warnings=warnings,
    )


def dictionary(df: pd.DataFrame) -> str:
    """A plain description of every column, for prompts: meaning, levels, counts."""
    lines = [f"Cohort: {df['cohort'].iloc[0]}. {len(df)} patients; one row per patient."]
    extra = {c: "a column from the uploaded data" for c in df.columns if c not in COLUMNS and c != "cohort"}
    for col, meaning in {**COLUMNS, **extra}.items():
        if col not in df.columns or df[col].isna().all():
            continue
        s = df[col].dropna()
        if col in NUMERIC or col in ("os_event", "tp53_mutated") or pd.api.types.is_numeric_dtype(s):
            desc = f"numeric, median {s.median():.3g}, range {s.min():.3g} to {s.max():.3g}"
            if s.nunique() <= 3:
                desc = "levels " + ", ".join(f"{v:g} (n={c})" for v, c in s.value_counts().items())
        else:
            levels = s.astype(str).value_counts()
            desc = "levels " + ", ".join(f"{v} (n={c})" for v, c in levels.head(12).items())
        lines.append(f"- {col}: {meaning}. {desc}; missing {int(df[col].isna().sum())}.")
    return "\n".join(lines)


def load(cohort: str, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Fetch (cached) and tidy one cohort. Patients without survival are kept."""
    if cohort not in STUDIES:
        raise ValueError(f"Unknown cohort {cohort!r}; choose from {sorted(STUDIES)}")
    return tidy(cohort, fetch_raw(cohort, cache_dir))
