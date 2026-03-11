# -*- coding: utf-8 -*-
import os
import json
import hashlib
from typing import Dict, List, Set, Tuple, Any

def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_raw_extraction(path: str, pmid: str, round_id: int, rec: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "pmid": str(pmid),
            "query_round": int(round_id),
            "extraction": rec
        }, ensure_ascii=False) + "\n")

def export_disease_nodes(triples: Set[Tuple[str, str, str]]) -> Dict[str, Dict[str, List[str]]]:
    nodes: Dict[str, Dict[str, List[str]]] = {}
    for h, r, t in triples:
        nodes.setdefault(h, {}).setdefault(r, []).append(t)
    for h in nodes:
        for r in nodes[h]:
            nodes[h][r] = sorted(set(nodes[h][r]))
    return nodes

def export_feature_index(triples: Set[Tuple[str, str, str]]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for h, r, t in triples:
        if r.startswith("HAS_"):
            index.setdefault(t, []).append(h)
    for t in index:
        index[t] = sorted(set(index[t]))
    return index

# -------- abstract dedup --------
def _normalize_for_hash(text: str) -> str:
    t = text.lower().strip()
    t = " ".join(t.split())
    return t

def hash_abstract(text: str) -> str:
    norm = _normalize_for_hash(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

def load_seen_hashes(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        if isinstance(arr, list):
            return set(str(x) for x in arr)
    except Exception:
        pass
    return set()

def save_seen_hashes(path: str, seen: Set[str]):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted(list(seen)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to save seen hashes: {e}")
