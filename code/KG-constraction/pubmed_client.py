# -*- coding: utf-8 -*-
import time
import random
from typing import List, Dict, Any
from Bio import Entrez

from .config import ENTREZ_EMAIL

Entrez.email = ENTREZ_EMAIL

def pubmed_search(query: str, retmax: int = 200, retries: int = 3) -> List[str]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with Entrez.esearch(db="pubmed", term=query, retmax=retmax) as handle:
                record = Entrez.read(handle)
            return record.get("IdList", [])
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 10) + random.random()
            print(f"[WARN] Entrez.esearch failed (attempt {attempt}/{retries}): {e} | sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)
    print(f"[ERROR] Entrez.esearch failed after {retries} retries: {last_err}")
    return []

def pubmed_fetch(pmids: List[str], retries: int = 3) -> List[Dict[str, Any]]:
    if not pmids:
        return []

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with Entrez.efetch(db="pubmed", id=",".join(pmids), retmode="xml") as handle:
                records = Entrez.read(handle)
            return records.get("PubmedArticle", [])
        except Exception as e:
            last_err = e
            sleep_s = min(2 ** attempt, 10) + random.random()
            print(f"[WARN] Entrez.efetch failed (attempt {attempt}/{retries}): {e} | sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    print(f"[ERROR] Entrez.efetch failed after {retries} retries: {last_err}")
    return []
