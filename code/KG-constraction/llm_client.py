# -*- coding: utf-8 -*-
import json
import requests
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlsplit, urlunsplit, quote

from .config import API_KEY, API_URL, MODEL_NAME

SCHEMA_TEXT = r'''
'''.strip()

def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=3,
        status=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s = requests.Session()
    # avoid env proxy/certs that can trigger latin-1 encoding errors
    s.trust_env = False
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = build_session()

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t

def _latin1_ok_headers(headers: Dict[str, str]) -> None:
    for k, v in headers.items():
        str(v).encode("latin-1")  # raise if cannot

def _normalize_proxy_url(proxy_url: str) -> str:
    parts = urlsplit(proxy_url.strip())
    if not parts.scheme or not parts.netloc:
        return proxy_url.strip()

    userinfo, _, hostport = parts.netloc.rpartition("@")
    host, _, port = hostport.partition(":")

    try:
        host_ascii = host.encode("idna").decode("ascii")
    except Exception:
        host_ascii = host

    userinfo_enc = quote(userinfo, safe=":@") if userinfo else ""
    netloc = f"{userinfo_enc}@{host_ascii}" if userinfo_enc else host_ascii
    if port:
        netloc = f"{netloc}:{port}"

    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&?/%")
    frag = quote(parts.fragment, safe="")
    return urlunsplit((parts.scheme, netloc, path, query, frag))

def _get_proxies_from_env() -> Optional[Dict[str, str]]:
    import os
    http_p = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_p = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if not http_p and not https_p:
        return None
    proxies: Dict[str, str] = {}
    if http_p:
        proxies["http"] = _normalize_proxy_url(http_p)
    if https_p:
        proxies["https"] = _normalize_proxy_url(https_p)
    return proxies

def llm_extract(abstract: str) -> Dict[str, Any]:
    if not API_KEY or API_KEY == "XXX":
        print("[ERROR] API_KEY is not set. Please export YUNWU_API_KEY.")
        return {}

    prompt = f"""
You are a pathology knowledge extraction system.
Extract ONLY explicitly stated information in the abstract (no guess, no general medical knowledge).
Return JSON strictly in the schema (no extra keys):

Schema:
{SCHEMA_TEXT}

Rules:
- Keep disease.name as the core disease name (e.g., "lung squamous cell carcinoma") if possible.
- Put modifiers like "metastatic" or "EGFR-mutated" into disease.qualifiers ONLY if explicitly stated.
- biomarkers.molecular_alterations.type must be in ["mutation","fusion","amplification","deletion","rearrangement","other"].

Confidence rubric (MUST follow):
- 0.95~1.00: explicitly stated in a direct sentence, no hedging, evidence_span is a verbatim short quote.
- 0.80~0.94: explicit but slightly paraphrased or requires combining two nearby sentences.
- 0.60~0.79: partially stated / implied (e.g., "associated with", "correlated with"), or not exact wording.
- 0.40~0.59: weakly implied or unclear disease-specificity.
- 0.00~0.39: speculative or not clearly supported — DO NOT output the item.
Hard rule: Do NOT output confidence=1.0 unless directly and unambiguously stated.

Abstract:
\"\"\"{abstract}\"\"\"
""".strip()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json; charset=utf-8",
    }
    _latin1_ok_headers(headers)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    proxies = _get_proxies_from_env()

    try:
        resp = SESSION.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=(10, 180),
            proxies=proxies,
        )

        if resp.status_code >= 400:
            print(f"[WARN] LLM HTTP {resp.status_code}: {resp.text[:200]}")
            return {}

        data = resp.json()
        text = _strip_code_fences(data["choices"][0]["message"]["content"])
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            print(f"[WARN] JSON parse failed. Raw head: {text[:200]}")
            return {}

    except UnicodeEncodeError as e:
        print("[ERROR] UnicodeEncodeError while preparing/sending request.")
        print(f"        error: {e}")
        print(f"        API_URL: {API_URL!r}")
        print(f"        proxies: {proxies!r}")
        head = prompt[:140].encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")
        print(f"        prompt_head(escaped): {head}")
        return {}

    except requests.RequestException as e:
        print(f"[WARN] LLM request failed: {type(e).__name__}: {e}")
        return {}
    except Exception as e:
        print(f"[WARN] Unexpected error in llm_extract: {type(e).__name__}: {e}")
        return {}
