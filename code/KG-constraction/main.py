# -*- coding: utf-8 -*-
import json
import time
import random

from .config import (
    RAW_DIR, KG_DIR, MEMORY_DIR,
    RAW_PATH, TRIPLES_PATH, EDGES_JSONL_PATH,
    DISEASE_NODES_PATH, FEATURE_INDEX_PATH,
    ENABLE_ABSTRACT_DEDUP, ABSTRACT_HASH_SEEN_PATH,
    MIN_CONFIDENCE,
)
from .pubmed_client import pubmed_search, pubmed_fetch
from .llm_client import llm_extract
from .kg_builder import PathologyKG
from .io_utils import (
    ensure_dirs, save_raw_extraction,
    export_disease_nodes, export_feature_index,
    load_seen_hashes, save_seen_hashes, hash_abstract
)

def run_deepsearch():
    ensure_dirs(RAW_DIR, KG_DIR, MEMORY_DIR)

    SEED_QUERY = """
    """.strip()

    print("[INFO] PubMed searching...")
    pmids = pubmed_search(SEED_QUERY, retmax=200)
    print(f"[INFO] PubMed got {len(pmids)} PMIDs")

    print("[INFO] PubMed fetching...")
    articles = pubmed_fetch(pmids)
    print(f"[INFO] Fetched {len(articles)} articles")

    kg = PathologyKG(min_confidence=MIN_CONFIDENCE)
    seen_hashes = load_seen_hashes(ABSTRACT_HASH_SEEN_PATH) if ENABLE_ABSTRACT_DEDUP else set()

    for i, article in enumerate(articles, 1):
        try:
            pmid = str(article["MedlineCitation"]["PMID"])
        except Exception:
            pmid = "UNKNOWN"

        abs_list = article.get("MedlineCitation", {}).get("Article", {}).get("Abstract", {}).get("AbstractText", [])
        if not abs_list:
            continue

        abstract = " ".join(str(x) for x in abs_list)

        if ENABLE_ABSTRACT_DEDUP:
            h = hash_abstract(abstract)
            if h in seen_hashes:
                print(f"[INFO] ({i}/{len(articles)}) PMID={pmid} skipped (duplicate abstract)")
                continue
            seen_hashes.add(h)

        print(f"[INFO] ({i}/{len(articles)}) PMID={pmid} extracting...")
        rec = llm_extract(abstract)

        # Save raw extraction regardless success for audit/retry
        save_raw_extraction(RAW_PATH, pmid, round_id=1, rec=rec)

        if rec:
            kg.add_record(pmid=pmid, rec=rec)

        time.sleep(0.8 + random.random() * 0.6)

    if ENABLE_ABSTRACT_DEDUP:
        save_seen_hashes(ABSTRACT_HASH_SEEN_PATH, seen_hashes)

    kg.save_triples(TRIPLES_PATH)
    kg.save_rich_edges(EDGES_JSONL_PATH)

    print(f"[INFO] Saved triples: {TRIPLES_PATH} (triples={len(kg.triples)})")
    print(f"[INFO] Saved rich edges: {EDGES_JSONL_PATH} (edges={len(kg.rich_edges)})")

    disease_nodes = export_disease_nodes(kg.triples)
    feature_index = export_feature_index(kg.triples)

    with open(DISEASE_NODES_PATH, "w", encoding="utf-8") as f:
        json.dump(disease_nodes, f, indent=2, ensure_ascii=False)

    with open(FEATURE_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_index, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved memory: {DISEASE_NODES_PATH}, {FEATURE_INDEX_PATH}")
    print(f"[INFO] MIN_CONFIDENCE={MIN_CONFIDENCE}, DEDUP={ENABLE_ABSTRACT_DEDUP}")

if __name__ == "__main__":
    run_deepsearch()
