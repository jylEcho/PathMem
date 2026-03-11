# -*- coding: utf-8 -*-
import os
import sys

# ---------- Encoding hardening ----------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

# ---------- NCBI ----------
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
# ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY", "")

# ---------- Paths ----------
BASE_DIR = os.getenv("BASE_DIR", "")
RAW_DIR = os.path.join(BASE_DIR, "raw_extractions")
KG_DIR = os.path.join(BASE_DIR, "kg")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")

RAW_PATH = os.path.join(RAW_DIR, "pubmed_round_1.jsonl")
TRIPLES_PATH = os.path.join(KG_DIR, "triples.tsv")
EDGES_JSONL_PATH = os.path.join(KG_DIR, "edges.jsonl")
DISEASE_NODES_PATH = os.path.join(MEMORY_DIR, "disease_nodes.json")
FEATURE_INDEX_PATH = os.path.join(MEMORY_DIR, "feature_index.json")

ENABLE_ABSTRACT_DEDUP = os.getenv("ENABLE_ABSTRACT_DEDUP", "1") == "1"
ABSTRACT_HASH_SEEN_PATH = os.path.join(MEMORY_DIR, "abstract_hash_seen.json")

# ---------- Quality controls ----------
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", ""))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", ""))

# Confidence calibration knobs
CONF_CAP_FOR_1 = float(os.getenv("CONF_CAP_FOR_1", ""))
CONF_CAP_HEDGE = float(os.getenv("CONF_CAP_HEDGE", ""))
CONF_CAP_SPECULATIVE = float(os.getenv("CONF_CAP_SPECULATIVE", ""))
CONF_CAP_EXPERIMENT = float(os.getenv("CONF_CAP_EXPERIMENT", ""))

FILTER_DRUG_EXPERIMENT_CLUES = os.getenv("FILTER_DRUG_EXPERIMENT_CLUES", "1") == "1"

# ---------- LLM API ----------
API_KEY = os.getenv("YUNWU_API_KEY", "")
API_URL = os.getenv("YUNWU_API_URL", "")
MODEL_NAME = os.getenv("YUNWU_MODEL_NAME", "")
