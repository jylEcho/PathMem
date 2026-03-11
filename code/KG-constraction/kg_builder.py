# -*- coding: utf-8 -*-
import re
import json
from typing import Any, Dict, List, Set, Tuple

from .config import (
    MAX_EVIDENCE_CHARS, MIN_CONFIDENCE,
    CONF_CAP_FOR_1, CONF_CAP_HEDGE, CONF_CAP_SPECULATIVE, CONF_CAP_EXPERIMENT,
    FILTER_DRUG_EXPERIMENT_CLUES,
)

def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().split())

def _clip_evidence(s: str) -> str:
    s = _norm_text(s)
    if len(s) <= MAX_EVIDENCE_CHARS:
        return s
    return s[:MAX_EVIDENCE_CHARS].rstrip() + "…"

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v < 0:
            return 0.0
        if v > 1:
            return 1.0
        return v
    except Exception:
        return default

# -------- confidence calibration --------
HEDGE_PAT = re.compile(r"\b(suggest|suggests|suggested|may|might|potential|likely|indicat\w*|associate\w*|correlat\w*)\b", re.I)
SPEC_PAT = re.compile(r"\b(hypothes\w*|propos\w*|possibly|could|unclear)\b", re.I)
EXPERIMENT_PAT = re.compile(r"\b(in vitro|in vivo|xenograft|cell line|treated|treatment|therapy|drug|inhibit\w*|tumor volume|tumour volume)\b", re.I)

def calibrate_confidence(raw_conf: Any, evidence: str) -> float:
    c = _safe_float(raw_conf, default=0.0)
    ev = (evidence or "")

    if c >= 0.999:
        c = min(c, CONF_CAP_FOR_1)
    if HEDGE_PAT.search(ev):
        c = min(c, CONF_CAP_HEDGE)
    if SPEC_PAT.search(ev):
        c = min(c, CONF_CAP_SPECULATIVE)
    if EXPERIMENT_PAT.search(ev):
        c = min(c, CONF_CAP_EXPERIMENT)
    return max(0.0, min(1.0, c))

# -------- relation slugging --------
REL_CLEAN_PAT = re.compile(r"[^A-Z0-9_]+")

def slug_relation(rel: str) -> str:
    r = _norm_text(rel).upper().replace(" ", "_")
    r = REL_CLEAN_PAT.sub("_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    return r

# -------- molecular type normalization --------
ALLOWED_MOL_TYPES = {"mutation", "fusion", "amplification", "deletion", "rearrangement", "other"}

def normalize_mol_type(typ: str) -> str:
    t = _norm_text(typ).lower()
    if not t:
        return "other"
    if t in ALLOWED_MOL_TYPES:
        return t
    if "copy number" in t or t in {"cnv", "copy_number_variation", "copy-number variation"}:
        return "other"
    if "methyl" in t:
        return "other"
    if "snv" in t or "variant" in t:
        return "mutation"
    return "other"

# -------- clue filtering --------
CLUE_FILTER_PAT = re.compile(
    r"\b(in vitro|in vivo|xenograft|cell line|crm197|cetuximab|afatinib|treated|treatment|therapy|drug|inhibit|tumor volume|tumour volume)\b",
    re.I
)

def clue_is_allowed(clue: str, evidence: str) -> bool:
    if not FILTER_DRUG_EXPERIMENT_CLUES:
        return True
    txt = f"{clue} {evidence}"
    return CLUE_FILTER_PAT.search(txt) is None

class PathologyKG:
    def __init__(self, min_confidence: float = MIN_CONFIDENCE):
        self.min_confidence = float(min_confidence)
        self.triples: Set[Tuple[str, str, str]] = set()
        self.rich_edges: List[Dict[str, Any]] = []

    def _add_edge(self, h: str, r: str, t: str, pmid: str,
                  evidence: str = "", confidence: float = 0.0):
        h = _norm_text(h)
        r = slug_relation(r)
        t = _norm_text(t)
        if not h or not r or not t:
            return

        conf = _safe_float(confidence, default=0.0)
        if conf < self.min_confidence:
            return

        ev = _clip_evidence(evidence) if evidence else ""
        self.triples.add((h, r, t))
        self.rich_edges.append({
            "head": h,
            "relation": r,
            "tail": t,
            "pmid": str(pmid),
            "confidence": conf,
            "evidence_span": ev
        })

    def add_record(self, pmid: str, rec: Dict[str, Any]):
        if not rec or not isinstance(rec, dict):
            return

        disease = rec.get("disease", {}) or {}
        disease_name = _norm_text(disease.get("name", ""))
        if not disease_name:
            return

        qualifiers = disease.get("qualifiers", []) or []
        for q in qualifiers:
            if isinstance(q, str) and _norm_text(q):
                self._add_edge(disease_name, "HAS_QUALIFIER", _norm_text(q), pmid, confidence=0.85)

        sites = rec.get("sites", {}) or {}
        primary_site = _norm_text(sites.get("primary_site", ""))
        specimen_site = _norm_text(sites.get("specimen_site", ""))
        metastatic_sites = sites.get("metastatic_sites", []) or []

        if primary_site:
            self._add_edge(disease_name, "PRIMARY_SITE", primary_site, pmid, confidence=0.85)
        if specimen_site:
            self._add_edge(disease_name, "SPECIMEN_SITE", specimen_site, pmid, confidence=0.85)
        for ms in metastatic_sites:
            if isinstance(ms, str) and _norm_text(ms):
                self._add_edge(disease_name, "METASTASIS_TO", _norm_text(ms), pmid, confidence=0.85)

        hist = rec.get("histology", {}) or {}
        histologic_type = _norm_text(hist.get("histologic_type", ""))
        grade = _norm_text(hist.get("grade", ""))

        if histologic_type:
            self._add_edge(disease_name, "HAS_HISTOLOGIC_TYPE", histologic_type, pmid, confidence=0.85)
        if grade:
            self._add_edge(disease_name, "HAS_GRADE", grade, pmid, confidence=0.85)

        feats = rec.get("features", {}) or {}
        for x in feats.get("architectural_patterns", []) or []:
            if isinstance(x, str) and _norm_text(x):
                self._add_edge(disease_name, "HAS_ARCHITECTURE", _norm_text(x), pmid, confidence=0.80)
        for x in feats.get("cellular_features", []) or []:
            if isinstance(x, str) and _norm_text(x):
                self._add_edge(disease_name, "HAS_CELLULAR_FEATURE", _norm_text(x), pmid, confidence=0.80)
        for x in feats.get("morphologic_features", []) or []:
            if isinstance(x, str) and _norm_text(x):
                self._add_edge(disease_name, "HAS_MORPHOLOGY", _norm_text(x), pmid, confidence=0.80)

        b = rec.get("biomarkers", {}) or {}

        for item in b.get("ihc_markers", []) or []:
            if not isinstance(item, dict):
                continue
            marker = _norm_text(item.get("marker", ""))
            sop = _norm_text(item.get("status_or_pattern", ""))
            evidence = item.get("evidence_span", "") or ""
            conf = calibrate_confidence(item.get("confidence", 0.0), evidence)

            if marker and sop:
                self._add_edge(disease_name, "HAS_IHC_MARKER", f"{marker} | {sop}", pmid, evidence=evidence, confidence=conf)
            elif marker:
                self._add_edge(disease_name, "HAS_IHC_MARKER", marker, pmid, evidence=evidence, confidence=conf)

        for item in b.get("expression_markers", []) or []:
            if not isinstance(item, dict):
                continue
            marker = _norm_text(item.get("marker", ""))
            measure = _norm_text(item.get("measure", ""))
            value = _norm_text(item.get("value", ""))
            evidence = item.get("evidence_span", "") or ""
            conf = calibrate_confidence(item.get("confidence", 0.0), evidence)

            if marker and measure and value:
                self._add_edge(disease_name, "HAS_EXPRESSION", f"{marker} | {measure}={value}", pmid, evidence=evidence, confidence=conf)
            elif marker and value:
                self._add_edge(disease_name, "HAS_EXPRESSION", f"{marker} | {value}", pmid, evidence=evidence, confidence=conf)
            elif marker:
                self._add_edge(disease_name, "HAS_EXPRESSION", marker, pmid, evidence=evidence, confidence=conf)

        for item in b.get("molecular_alterations", []) or []:
            if not isinstance(item, dict):
                continue
            raw_typ = _norm_text(item.get("type", ""))
            typ = normalize_mol_type(raw_typ)
            gene = _norm_text(item.get("gene", ""))
            alt = _norm_text(item.get("alteration", ""))
            evidence = item.get("evidence_span", "") or ""
            conf = calibrate_confidence(item.get("confidence", 0.0), evidence)

            if raw_typ and typ == "other" and raw_typ.lower() not in ALLOWED_MOL_TYPES:
                alt = f"{alt} (type:{raw_typ})" if alt else f"type:{raw_typ}"

            rel = f"HAS_{typ.upper()}" if typ else "HAS_MOLECULAR_ALTERATION"

            if gene and alt:
                self._add_edge(disease_name, rel, f"{gene} | {alt}", pmid, evidence=evidence, confidence=conf)
            elif gene:
                self._add_edge(disease_name, rel, gene, pmid, evidence=evidence, confidence=conf)

        for item in b.get("serum_markers", []) or []:
            if not isinstance(item, dict):
                continue
            marker = _norm_text(item.get("marker", ""))
            sov = _norm_text(item.get("status_or_value", ""))
            evidence = item.get("evidence_span", "") or ""
            conf = calibrate_confidence(item.get("confidence", 0.0), evidence)

            if marker and sov:
                self._add_edge(disease_name, "HAS_SERUM_MARKER", f"{marker} | {sov}", pmid, evidence=evidence, confidence=conf)
            elif marker:
                self._add_edge(disease_name, "HAS_SERUM_MARKER", marker, pmid, evidence=evidence, confidence=conf)

        for item in rec.get("diagnostic_clues", []) or []:
            if isinstance(item, dict):
                clue = _norm_text(item.get("clue", ""))
                evidence = item.get("evidence_span", "") or ""
                conf = calibrate_confidence(item.get("confidence", 0.0), evidence)
                if clue and clue_is_allowed(clue, evidence):
                    self._add_edge(disease_name, "HAS_DIAGNOSTIC_CLUE", clue, pmid, evidence=evidence, confidence=conf)
            elif isinstance(item, str) and _norm_text(item):
                clue = _norm_text(item)
                if clue_is_allowed(clue, ""):
                    self._add_edge(disease_name, "HAS_DIAGNOSTIC_CLUE", clue, pmid, confidence=0.75)

    def save_triples(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for h, r, t in sorted(self.triples):
                f.write(f"{h}\t{r}\t{t}\n")

    def save_rich_edges(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for e in self.rich_edges:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
