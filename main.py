import asyncio
import json
import logging
import os
import re
from datetime import date
from html import unescape

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobmatch")

app = FastAPI(title="JobMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Greenhouse boards (direct company ATS pages)
# NOTE: These must be luxury/fashion/textile/consumer companies
# to matter for this app. The original list here was Israeli
# tech/fintech boards, which is why almost every job scored
# near-zero. Replace tokens below with real boards once you find
# them (check a company's careers page URL - if it's
# https://boards.greenhouse.io/{token}, that token goes here).
# Left mostly empty on purpose rather than filled with irrelevant
# placeholders.
# ============================================================
GREENHOUSE_BOARDS: list[str] = [
    "poshmark",  # fashion resale marketplace, US
    # Add more once verified: try https://boards-api.greenhouse.io/v1/boards/{token}/jobs
    # in a browser — if it returns JSON with a "jobs" array, the token is valid.
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

# ============================================================
# Lever boards (a second common ATS, different API shape).
# LVMH, Kering, and Richemont run proprietary careers systems and
# are NOT on Greenhouse or Lever — no token exists for them. What
# IS commonly on Lever are fashion-tech / marketplace / DTC brands
# with genuine luxury-adjacent commercial roles.
# ============================================================
LEVER_BOARDS: list[str] = [
    "farfetch",  # luxury fashion e-commerce, global (incl. Asia offices)
    # Add more once verified: try https://api.lever.co/v0/postings/{token}?mode=json
    # in a browser — if it returns a JSON array (even empty), the token is valid.
]
LEVER_URL = "https://api.lever.co/v0/postings/{token}?mode=json"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_BATCH_SIZE = 25
GEMINI_CONCURRENCY = 5
GEMINI_DESCRIPTION_CHARS = 600

CANDIDATE_LANGUAGES = ["English", "French", "Italian", "German", "Spanish", "Hebrew (basic)"]

# Regions to exclude within France specifically (does not affect Switzerland's
# fr_CH locale or any other market, since it's matched against location text
# only for jobs sourced from a France-tagged source).
FRANCE_EXCLUDED_LOCATIONS = ["paris", "lyon"]

# Query expansion: for a commercial/luxury/textile CV like this one,
# a single literal role string under-returns. We broaden the search
# net with a handful of related titles, then let Gemini do the real
# filtering for fit.
ROLE_SYNONYMS = [
    "Commercial Director",
    "Sales Director",
    "General Manager",
    "Business Development Director",
    "Country Manager",
    "Managing Director",
    "Non-Executive Director",
    "Board Member",
    "Operating Partner",
    "Chief Commercial Officer",
]
INDUSTRY_KEYWORDS = ["luxury", "fashion", "textile", "private equity"]

# A few dedicated, hand-picked queries for board and PE-specific searches,
# added on top of the synonym x keyword cross-product below.
EXTRA_QUERIES = [
    "Non-Executive Director luxury fashion",
    "Board Advisor consumer goods",
    "Private Equity Portfolio Operating Partner",
    "Private Equity Commercial Due Diligence",
]


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _build_queries(role: str) -> list[str]:
    """Expand a single role string into several search queries to widen recall."""
    queries = {role.strip()}
    for synonym in ROLE_SYNONYMS:
        for keyword in INDUSTRY_KEYWORDS:
            queries.add(f"{synonym} {keyword}")
    queries.update(EXTRA_QUERIES)
    # Cap to keep the number of outbound API calls sane
    return list(queries)[:14]


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    response = await client.get(GREENHOUSE_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("name"),
                "apply_url": job.get("absolute_url"),
                "company": token,
                "description": _strip_html(job.get("content")),
                "source": "greenhouse",
            }
        )
    return jobs


async def fetch_lever(client: httpx.AsyncClient, token: str) -> list[dict]:
    response = await client.get(LEVER_URL.format(token=token))
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data:
        categories = job.get("categories") or {}
        jobs.append(
            {
                "title": job.get("text"),
                "location": categories.get("location"),
                "apply_url": job.get("hostedUrl"),
                "company": token,
                "description": _strip_html(job.get("descriptionPlain") or job.get("description")),
                "source": "lever",
            }
        )
    return jobs


# --- Adzuna (broad country coverage, free tier) ---
ADZUNA_COUNTRIES = ["us", "gb", "it", "fr", "de", "il"]  # "il" is best-effort — Adzuna's Israel coverage is unconfirmed, but the fetcher fails gracefully if unsupported. "ae" removed - confirmed 404, Adzuna doesn't cover the UAE.
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


async def fetch_adzuna(client: httpx.AsyncClient, country: str, query: str, where: str = "") -> list[dict]:
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not (app_id and app_key):
        return []
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 50 if country == "us" else 20,  # more volume in the US, as requested
        "what": query,
        "content-type": "application/json",
    }
    if where:
        params["where"] = where
    response = await client.get(ADZUNA_URL.format(country=country), params=params)
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("results", []):
        jobs.append(
            {
                "title": job.get("title"),
                "location": (job.get("location") or {}).get("display_name"),
                "apply_url": job.get("redirect_url"),
                "company": (job.get("company") or {}).get("display_name") or "Unknown",
                "description": _strip_html(job.get("description")),
                "source": f"adzuna_{country}" + (f"_{where}" if where else ""),
            }
        )
    return jobs


# --- Reed (UK) ---
REED_URL = "https://www.reed.co.uk/api/1.0/search"


async def fetch_reed(client: httpx.AsyncClient, query: str) -> list[dict]:
    api_key = os.environ.get("REED_API_KEY")
    if not api_key:
        return []
    response = await client.get(
        REED_URL,
        params={"keywords": query, "resultsToTake": 20},
        auth=(api_key, ""),
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("results", []):
        jobs.append(
            {
                "title": job.get("jobTitle"),
                "location": job.get("locationName"),
                "apply_url": job.get("jobUrl"),
                "company": job.get("employerName") or "Unknown",
                "description": _strip_html(job.get("jobDescription")),
                "source": "reed",
            }
        )
    return jobs


# --- Careerjet (broad locale coverage — verify each locale is supported
# for your affiliate account at careerjet.com/partners/api) ---
CAREERJET_URL = "https://public.api.careerjet.net/search"
CAREERJET_LOCALES = [
    "en_US",  # USA
    "en_GB",  # fallback / UK
    "it_IT",  # Italy
    "fr_FR",  # France (candidate speaks French)
    "de_DE",  # Germany (candidate speaks German)
    "fr_CH",  # Switzerland (French-speaking) — Geneva luxury/watch hub
    "de_CH",  # Switzerland (German-speaking) — Zurich
    "tr_TR",  # Turkey
    "ar_AE",  # UAE
    "en_IL",  # Israel — best-effort, unconfirmed Careerjet support, fails gracefully
]


async def fetch_careerjet(client: httpx.AsyncClient, query: str, locale: str) -> list[dict]:
    affid = os.environ.get("CAREERJET_AFFILIATE_ID", "")
    try:
        response = await client.get(
            CAREERJET_URL,
            params={
                "keywords": query,
                "locale_code": locale,
                "affid": affid,
                "pagesize": 20,
            },
        )
        response.raise_for_status()
    except httpx.HTTPError:
        # Some locales may not be supported, may be rate-limited without an
        # affid, or the host may be unreachable from this network (e.g. cloud
        # provider IP ranges blocked) - degrade gracefully either way
        return []
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        jobs.append(
            {
                "title": job.get("title"),
                "location": job.get("locations"),
                "apply_url": job.get("url"),
                "company": job.get("company") or "Unknown",
                "description": _strip_html(job.get("description")),
                "source": f"careerjet_{locale}",
            }
        )
    return jobs


# --- Jooble (broadest geographic coverage, incl. many Asian/CIS markets) ---
JOOBLE_URL = "https://jooble.org/api/{key}"


async def fetch_jooble(client: httpx.AsyncClient, query: str, location: str = "") -> list[dict]:
    api_key = os.environ.get("JOOBLE_API_KEY")
    if not api_key:
        return []
    response = await client.post(
        JOOBLE_URL.format(key=api_key),
        json={"keywords": query, "location": location},
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        jobs.append(
            {
                "title": job.get("title"),
                "location": job.get("location"),
                "apply_url": job.get("link"),
                "company": job.get("company") or "Unknown",
                "description": _strip_html(job.get("snippet")),
                "source": f"jooble_{location}" if location else "jooble",
            }
        )
    return jobs


# --- HeadHunter Kazakhstan (hh.kz) — keyless public API ---
HH_KZ_URL = "https://api.hh.ru/vacancies"
HH_KZ_AREA_ID = 40  # Kazakhstan


async def fetch_hh_kz(client: httpx.AsyncClient, query: str) -> list[dict]:
    try:
        response = await client.get(
            HH_KZ_URL,
            params={"text": query, "area": HH_KZ_AREA_ID, "per_page": 20},
            headers={"User-Agent": "JobMatchApp/1.0 (contact@example.com)"},
        )
        response.raise_for_status()
    except httpx.HTTPError:
        # hh.ru appears to reject requests from cloud/datacenter IP ranges
        # regardless of headers (observed 400/403 from multiple hosting
        # providers) - this is outside our control, so degrade gracefully
        # like the other best-effort sources rather than surfacing an error.
        return []
    data = response.json()
    jobs = []
    for item in data.get("items", []):
        snippet = item.get("snippet") or {}
        jobs.append(
            {
                "title": item.get("name"),
                "location": (item.get("area") or {}).get("name"),
                "apply_url": item.get("alternate_url"),
                "company": (item.get("employer") or {}).get("name") or "Unknown",
                "description": _strip_html(snippet.get("responsibility", "")),
                "source": "hh_kz",
            }
        )
    return jobs


async def _fetch_all_jobs(role: str = "") -> list[dict]:
    source_semaphore = asyncio.Semaphore(20)  # cap concurrent outbound calls across all sources

    async def _throttled(coro):
        async with source_semaphore:
            return await coro

    async with httpx.AsyncClient(timeout=45.0) as client:
        tasks: list = [fetch_board(client, token) for token in GREENHOUSE_BOARDS]
        tasks += [fetch_lever(client, token) for token in LEVER_BOARDS]

        if role:
            queries = _build_queries(role)
            for query in queries:
                for country in ADZUNA_COUNTRIES:
                    tasks.append(fetch_adzuna(client, country, query))
                tasks.append(fetch_reed(client, query))
                tasks.append(fetch_jooble(client, query))
                tasks.append(fetch_jooble(client, query, location="Israel"))
                tasks.append(fetch_hh_kz(client, query))
                for locale in CAREERJET_LOCALES:
                    tasks.append(fetch_careerjet(client, query, locale))

        results = await asyncio.gather(*(_throttled(t) for t in tasks), return_exceptions=True)

    jobs: list[dict] = []
    failures = 0
    for result in results:
        if isinstance(result, Exception):
            failures += 1
            logger.warning("A job source failed: %s", result)
            continue
        jobs.extend(result)

    logger.info("Fetched %d jobs total (%d source calls failed/skipped)", len(jobs), failures)

    # Exclude Paris and Lyon specifically within France-sourced results
    def _is_excluded_france_job(job: dict) -> bool:
        source = (job.get("source") or "").lower()
        if not (source.startswith("adzuna_fr") or source == "careerjet_fr_fr"):
            return False
        location = (job.get("location") or "").lower()
        return any(excluded in location for excluded in FRANCE_EXCLUDED_LOCATIONS)

    jobs = [job for job in jobs if not _is_excluded_france_job(job)]

    # De-duplicate by (title, company) since query expansion causes overlap
    seen = set()
    deduped = []
    for job in jobs:
        key = (job.get("title"), job.get("company"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)

    return deduped


@app.get("/jobs")
async def get_jobs(role: str = "") -> dict:
    jobs = await _fetch_all_jobs(role=role)
    public_jobs = [{k: v for k, v in job.items() if k != "description"} for job in jobs]
    return {"count": len(public_jobs), "jobs": public_jobs}


@app.get("/debug/sources")
async def debug_sources(role: str = "Commercial Director luxury fashion") -> dict:
    """Diagnostic endpoint: shows how many jobs each source returns for a given role."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        checks = {
            "adzuna_us": fetch_adzuna(client, "us", role),
            "adzuna_gb": fetch_adzuna(client, "gb", role),
            "adzuna_it": fetch_adzuna(client, "it", role),
            "adzuna_fr": fetch_adzuna(client, "fr", role),
            "reed": fetch_reed(client, role),
            "jooble": fetch_jooble(client, role),
            "hh_kz": fetch_hh_kz(client, role),
            "careerjet_it_IT": fetch_careerjet(client, role, "it_IT"),
        }
        for token in GREENHOUSE_BOARDS:
            checks[f"greenhouse_{token}"] = fetch_board(client, token)
        for token in LEVER_BOARDS:
            checks[f"lever_{token}"] = fetch_lever(client, token)
        results = await asyncio.gather(*checks.values(), return_exceptions=True)

    summary = {}
    for name, result in zip(checks.keys(), results):
        if isinstance(result, Exception):
            summary[name] = f"ERROR: {result}"
        else:
            summary[name] = len(result)

    # Adzuna/Reed/Jooble parse "results"/"jobs" out of the response body and
    # silently return [] if that key is missing - which means an API error
    # disguised as a normal-looking response (bad key, quota exceeded, etc.)
    # would otherwise look identical to "genuinely zero results". Show the
    # raw response for these three so that distinction is visible.
    raw = {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        app_id = os.environ.get("ADZUNA_APP_ID")
        app_key = os.environ.get("ADZUNA_APP_KEY")
        if app_id and app_key:
            try:
                r = await client.get(
                    ADZUNA_URL.format(country="us"),
                    params={
                        "app_id": app_id,
                        "app_key": app_key,
                        "what": role,
                        "results_per_page": 1,
                        "content-type": "application/json",
                    },
                )
                raw["adzuna_us_raw"] = {"status": r.status_code, "body": r.text[:500]}
            except Exception as exc:
                raw["adzuna_us_raw"] = f"EXCEPTION: {exc}"

        reed_key = os.environ.get("REED_API_KEY")
        if reed_key:
            try:
                r = await client.get(
                    REED_URL, params={"keywords": role, "resultsToTake": 1}, auth=(reed_key, "")
                )
                raw["reed_raw"] = {"status": r.status_code, "body": r.text[:500]}
            except Exception as exc:
                raw["reed_raw"] = f"EXCEPTION: {exc}"

        jooble_key = os.environ.get("JOOBLE_API_KEY")
        if jooble_key:
            try:
                r = await client.post(JOOBLE_URL.format(key=jooble_key), json={"keywords": role})
                raw["jooble_raw"] = {"status": r.status_code, "body": r.text[:500]}
            except Exception as exc:
                raw["jooble_raw"] = f"EXCEPTION: {exc}"

    summary["_raw_previews"] = raw
    return summary


class RankRequest(BaseModel):
    cv: str = Field(..., min_length=1, description="Plain text of the candidate's CV")
    role: str = Field(..., min_length=1, description="The role the candidate wants")


def _build_prompt(cv: str, role: str, batch: list[dict]) -> str:
    listing = [
        {
            "index": i,
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "description": (job.get("description") or "")[:GEMINI_DESCRIPTION_CHARS],
        }
        for i, job in enumerate(batch)
    ]
    return (
        "You are a recruiting assistant. Score how well each job listing fits the "
        "candidate, given their CV and the role they want.\n\n"
        f"CANDIDATE CV:\n{cv}\n\n"
        f"TARGET ROLE:\n{role}\n\n"
        f"CANDIDATE LANGUAGES: {', '.join(CANDIDATE_LANGUAGES)}\n\n"
        f"JOBS:\n{json.dumps(listing)}\n\n"
        "For every job in JOBS, return its index, a fit score from 0 to 100 "
        "(100 = perfect fit), and a short one-sentence reason for the score that "
        "references the candidate's CV and/or target role.\n\n"
        "LANGUAGE RULE: If a job description explicitly requires fluency in a "
        "language NOT in the candidate's language list (e.g. Mandarin, Cantonese, "
        "Thai, Vietnamese, Kazakh, Turkish, Uzbek) as a mandatory qualification, "
        "score that job 0 and state the language mismatch as the reason. Do not "
        "penalize jobs that are silent on language requirements or that only "
        "mention local language as 'a plus' rather than mandatory.\n\n"
        "RELEVANCE RULE: This candidate is a senior commercial/strategy leader in "
        "luxury fashion and textiles, not a generalist. Score roles outside "
        "commercial leadership, general management, business development, or "
        "luxury/fashion/textile/retail operations low (under 30), even if the "
        "job title superficially contains a matching keyword."
    )


_GEMINI_RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {"type": "INTEGER"},
            "score": {"type": "INTEGER"},
            "reason": {"type": "STRING"},
        },
        "required": ["index", "score", "reason"],
    },
}


async def _score_batch(
    client: httpx.AsyncClient,
    api_key: str,
    cv: str,
    role: str,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    payload = {
        "contents": [{"parts": [{"text": _build_prompt(cv, role, batch)}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": _GEMINI_RESPONSE_SCHEMA,
        },
    }

    async with semaphore:
        response = await client.post(GEMINI_URL, params={"key": api_key}, json=payload)
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    results = json.loads(text)

    scored = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < len(batch)):
            continue
        job = batch[index]
        scored.append(
            {
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "apply_url": job.get("apply_url"),
                "score": item.get("score", 0),
                "reason": item.get("reason", ""),
            }
        )
    return scored


async def _score_jobs_with_gemini(jobs: list[dict], cv: str, role: str) -> list[dict]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set")

    batches = [jobs[i : i + GEMINI_BATCH_SIZE] for i in range(0, len(jobs), GEMINI_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(
            *(_score_batch(client, api_key, cv, role, batch, semaphore) for batch in batches),
            return_exceptions=True,
        )

    scored_jobs: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("A Gemini scoring batch failed: %s", result)
            continue
        scored_jobs.extend(result)
    return scored_jobs


@app.post("/rank")
async def rank_jobs(payload: RankRequest) -> dict:
    jobs = await _fetch_all_jobs(role=payload.role)
    if not jobs:
        return {"count": 0, "jobs": [], "note": "No jobs fetched from any source — check /debug/sources"}

    scored_jobs = await _score_jobs_with_gemini(jobs, payload.cv, payload.role)
    scored_jobs = [j for j in scored_jobs if j["score"] > 0]  # drop language/relevance mismatches
    scored_jobs.sort(key=lambda job: job["score"], reverse=True)
    return {"count": len(scored_jobs), "jobs": scored_jobs}


import pathlib

RECRUITERS_PATH = pathlib.Path(__file__).parent / "recruiters.json"


@app.get("/recruiters")
async def get_recruiters() -> dict:
    if not RECRUITERS_PATH.exists():
        raise HTTPException(status_code=404, detail="recruiters.json not found next to main.py")
    return json.loads(RECRUITERS_PATH.read_text())


# ============================================================
# Job-alert scanning (fully separate from /rank and /recruiters).
#
# Periodically (or on demand via /scan-and-alert) searches major job
# platforms for senior Textile, Sales/Commercial, Board/NED, and
# Fashion-Tech-commercial roles worldwide, using SerpAPI's "site:" search
# instead of scraping (scraping LinkedIn/Indeed directly would violate
# their Terms of Service — SerpAPI legitimately indexes their public
# pages). A lightweight Gemini pass filters the raw search snippets down
# to real, current, senior postings, then a digest is emailed via
# SendGrid. See README.md for how to point a hosting platform's cron
# scheduler at POST /scan-and-alert.
# ============================================================

SERPAPI_URL = "https://serpapi.com/search"
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"

PLATFORM_SITES = [
    "linkedin.com/jobs",
    "indeed.com",
    "glassdoor.com",
    "monster.com",
    "bayt.com",
    "naukri.com",
    "jobstreet.com",
    "wellfound.com",  # formerly AngelList Talent — startup-heavy
    "builtin.com",  # startup/tech-heavy, under-indexed on general boards
    "suitex.it",  # Suitex International — fashion/luxury/design executive search, Milan/Venice
    "luxetalent.it",  # Luxe Talent's Italian site (as given)
    "luxetalent.net",  # Luxe Talent's main international hub — offices across Europe, worldwide placements
    "fashionjobs.com",  # global fashion/luxury/beauty job board, 50+ country subdomains (site: matches all of them)
    "businessoffashion.com",  # BoF Careers — international fashion industry jobs, business/exec-skewed
]

WATCH_QUERIES = [
    # Textile / Sales & Commercial / Board
    "Textile Director",
    "Textile Commercial Director",
    "VP Sales luxury",
    "Chief Commercial Officer fashion",
    "Sales Director luxury goods",
    "Non-Executive Director fashion",
    "Board Member textile",
    "Board Member luxury goods",
    # Fashion Tech / Textile Tech — commercial track only, early-stage
    # startups anywhere in the world. Deliberately excludes product,
    # engineering, design, and operations roles at these same companies.
    "Head of Sales fashion tech startup",
    "VP Business Development textile technology",
    "Chief Commercial Officer fashion tech",
    "Commercial Director textile innovation",
    "Head of Sales sustainable materials startup",
    "Business Development Director digital fashion",
    "Commercial Lead supply chain traceability fashion",
]

ALERT_CATEGORIES = ["Textile", "Sales & Commercial", "Board", "Fashion Tech"]

SCAN_CONCURRENCY = 10
SCAN_RESULTS_PER_QUERY = 20
SCAN_BATCH_SIZE = 30


async def search_job_platforms(client: httpx.AsyncClient, query: str, site: str, tbs: str = "") -> list[dict]:
    """Search one job platform for a query via SerpAPI's site: operator (not scraping).

    tbs is SerpAPI's time-filter param (e.g. "qdr:w" for the past week) - when
    set, freshness is enforced by the search itself rather than left to Gemini
    to guess from a snippet.
    """
    api_key = os.environ.get("SERPAPI_KEY")
    if not api_key:
        return []

    params = {
        "engine": "google",
        "q": f"site:{site} {query}",
        "api_key": api_key,
        "num": SCAN_RESULTS_PER_QUERY,
    }
    if tbs:
        params["tbs"] = tbs

    response = await client.get(SERPAPI_URL, params=params)
    response.raise_for_status()
    data = response.json()
    hits = []
    for item in data.get("organic_results", []):
        hits.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "platform": site,
            }
        )
    return hits


async def _run_platform_scan(tbs: str = "") -> list[dict]:
    semaphore = asyncio.Semaphore(SCAN_CONCURRENCY)

    async def _throttled(client: httpx.AsyncClient, query: str, site: str):
        async with semaphore:
            return await search_job_platforms(client, query, site, tbs=tbs)

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [_throttled(client, query, site) for query in WATCH_QUERIES for site in PLATFORM_SITES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    hits: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("A platform scan call failed: %s", result)
            continue
        hits.extend(result)

    # De-duplicate by link since the same posting often surfaces under
    # several overlapping queries
    seen = set()
    deduped = []
    for hit in hits:
        link = hit.get("link")
        if not link or link in seen:
            continue
        seen.add(link)
        deduped.append(hit)

    return deduped


def _build_scan_filter_prompt(batch: list[dict]) -> str:
    listing = [
        {
            "title": hit.get("title"),
            "snippet": hit.get("snippet"),
            "link": hit.get("link"),
            "platform": hit.get("platform"),
        }
        for hit in batch
    ]
    categories = ", ".join(ALERT_CATEGORIES)
    return (
        "You are screening raw Google search results (title + snippet + link) "
        "for a job alert digest. These are search snippets, not full job "
        "descriptions, so use judgement.\n\n"
        f"RESULTS:\n{json.dumps(listing)}\n\n"
        "Keep ONLY results that are clearly a real, currently open, senior-level "
        "job posting (not a news article, blog post, old/expired listing, "
        "junior/mid-level role, or an unrelated search result) that falls into "
        f"one of these categories: {categories}.\n\n"
        "Category definitions:\n"
        "- Textile: senior commercial, sales, or general-management roles at "
        "textile manufacturers, mills, or textile divisions of larger "
        "companies.\n"
        "- Sales & Commercial: VP/Director/Chief-level sales, commercial, or "
        "business development roles in luxury goods, fashion, or apparel.\n"
        "- Board: Non-Executive Director, Board Member, or Board Advisor "
        "positions in fashion, luxury, or textile companies.\n"
        "- Fashion Tech: COMMERCIAL-TRACK ROLES ONLY (Head of Sales, VP "
        "Business Development, Chief Commercial Officer, Commercial Director) "
        "at startups or scale-ups building technology for fashion, apparel, or "
        "textiles - e.g. smart/sustainable materials, textile recycling tech, "
        "3D/digital fashion design tools, supply chain traceability, "
        "AI-driven design or merchandising, fashion resale/rental platforms, "
        "made-to-measure/on-demand manufacturing tech. Do NOT include product, "
        "engineering, design, or operations roles in this category, even at "
        "the same companies.\n\n"
        "For each result you keep, return: title, your best guess at the "
        "company name (company_guess), the link, the platform, the category "
        f"(exactly one of {categories}), and a one-sentence reason it "
        "qualifies (one_line_reason). Drop everything else - do not return an "
        "entry for rejected results."
    )


_SCAN_FILTER_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "company_guess": {"type": "STRING"},
            "link": {"type": "STRING"},
            "platform": {"type": "STRING"},
            "category": {"type": "STRING", "enum": ALERT_CATEGORIES},
            "one_line_reason": {"type": "STRING"},
        },
        "required": ["title", "company_guess", "link", "platform", "category", "one_line_reason"],
    },
}


async def _filter_scan_batch(
    client: httpx.AsyncClient, api_key: str, batch: list[dict], semaphore: asyncio.Semaphore
) -> list[dict]:
    payload = {
        "contents": [{"parts": [{"text": _build_scan_filter_prompt(batch)}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": _SCAN_FILTER_SCHEMA,
        },
    }
    async with semaphore:
        response = await client.post(GEMINI_URL, params={"key": api_key}, json=payload)
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


async def _filter_scan_results(hits: list[dict]) -> list[dict]:
    if not hits:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set")

    batches = [hits[i : i + SCAN_BATCH_SIZE] for i in range(0, len(hits), SCAN_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(GEMINI_CONCURRENCY)

    async with httpx.AsyncClient(timeout=60.0) as client:
        results = await asyncio.gather(
            *(_filter_scan_batch(client, api_key, batch, semaphore) for batch in batches),
            return_exceptions=True,
        )

    filtered: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning("A scan filter batch failed: %s", result)
            continue
        filtered.extend(result)
    return filtered


def _build_digest_html(jobs: list[dict]) -> str:
    by_category: dict[str, list[dict]] = {}
    for job in jobs:
        by_category.setdefault(job.get("category", "Other"), []).append(job)

    sections = []
    for category in ALERT_CATEGORIES:
        category_jobs = by_category.get(category, [])
        if not category_jobs:
            continue
        items = "".join(
            f"<li><a href=\"{job['link']}\">{job['title']}</a> "
            f"&mdash; {job.get('company_guess', '')} ({job['platform']})<br>"
            f"<small>{job.get('one_line_reason', '')}</small></li>"
            for job in category_jobs
        )
        sections.append(f"<h2>{category}</h2><ul>{items}</ul>")

    if not sections:
        return "<p>No new matching roles found in this scan.</p>"
    return "".join(sections)


async def _send_digest_email(client: httpx.AsyncClient, jobs: list[dict]) -> None:
    api_key = os.environ.get("SENDGRID_API_KEY")
    to_email = os.environ.get("ALERT_EMAIL_TO")
    from_email = os.environ.get("ALERT_EMAIL_FROM")
    if not (api_key and to_email and from_email):
        raise HTTPException(
            status_code=500,
            detail="SENDGRID_API_KEY, ALERT_EMAIL_TO, and ALERT_EMAIL_FROM must all be set",
        )

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": f"Job alert digest — {len(jobs)} role(s) found",
        "content": [{"type": "text/html", "value": _build_digest_html(jobs)}],
    }
    response = await client.post(
        SENDGRID_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    response.raise_for_status()


@app.get("/scan-preview")
async def scan_preview() -> dict:
    """Run the search + filter pipeline and return matches without emailing anyone."""
    hits = await _run_platform_scan()
    jobs = await _filter_scan_results(hits)
    return {"raw_hits": len(hits), "matches": len(jobs), "jobs": jobs}


@app.post("/scan-and-alert")
async def scan_and_alert() -> dict:
    """Run the full pipeline (search -> filter -> email) and return a summary."""
    hits = await _run_platform_scan()
    jobs = await _filter_scan_results(hits)

    async with httpx.AsyncClient(timeout=30.0) as client:
        await _send_digest_email(client, jobs)

    return {"raw_hits": len(hits), "matches": len(jobs), "emailed": True, "jobs": jobs}


# ============================================================
# Weekly "Top 10 Fresh Jobs" (separate from /rank, /recruiters, and
# /scan-and-alert - reuses WATCH_QUERIES, PLATFORM_SITES,
# search_job_platforms, _run_platform_scan, and _filter_scan_results, but
# narrows the broad digest down to exactly the 10 most relevant, freshly
# published (past week) postings overall via a second Gemini ranking pass).
# ============================================================

_TOP10_RANK_SCHEMA = {
    "type": "ARRAY",
    "maxItems": 10,
    "items": {
        "type": "OBJECT",
        "properties": {
            "rank": {"type": "INTEGER"},
            "title": {"type": "STRING"},
            "company_guess": {"type": "STRING"},
            "link": {"type": "STRING"},
            "platform": {"type": "STRING"},
            "category": {"type": "STRING", "enum": ALERT_CATEGORIES},
            "reason": {"type": "STRING"},
        },
        "required": ["rank", "title", "company_guess", "link", "platform", "category", "reason"],
    },
}


def _build_top10_prompt(candidates: list[dict]) -> str:
    listing = [
        {
            "title": job.get("title"),
            "company_guess": job.get("company_guess"),
            "link": job.get("link"),
            "platform": job.get("platform"),
            "category": job.get("category"),
            "reason": job.get("one_line_reason"),
        }
        for job in candidates
    ]
    return (
        "From this list of job postings, select and rank the 10 most relevant "
        "and senior opportunities overall, prioritizing Textile, senior "
        "Sales/Commercial leadership, Board/Non-Executive Director roles, and "
        "commercial roles at fashion-tech or textile-tech startups. Return "
        "exactly 10 items (fewer only if there are truly fewer than 10 "
        "qualifying results), each with title, company_guess, link, platform, "
        "category, and a one-sentence reason this made the top 10.\n\n"
        f"CANDIDATES:\n{json.dumps(listing)}"
    )


async def _rank_top10(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set")

    payload = {
        "contents": [{"parts": [{"text": _build_top10_prompt(candidates)}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": _TOP10_RANK_SCHEMA,
        },
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GEMINI_URL, params={"key": api_key}, json=payload)
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
    results = json.loads(text)
    results.sort(key=lambda job: job.get("rank", 999))
    return results[:10]


def _build_top10_html(jobs: list[dict]) -> str:
    if not jobs:
        return "<p>No qualifying roles found this week.</p>"

    return "".join(
        f"<p>#{job['rank']}. <strong>{job['title']}</strong> at {job.get('company_guess', '')} "
        f"&mdash; {job['platform']}<br>"
        f"<a href=\"{job['link']}\">{job['link']}</a><br>"
        f"{job.get('reason', '')}</p>"
        for job in jobs
    )


async def _send_top10_email(client: httpx.AsyncClient, jobs: list[dict]) -> None:
    api_key = os.environ.get("SENDGRID_API_KEY")
    to_email = os.environ.get("ALERT_EMAIL_TO")
    from_email = os.environ.get("ALERT_EMAIL_FROM")
    if not (api_key and to_email and from_email):
        raise HTTPException(
            status_code=500,
            detail="SENDGRID_API_KEY, ALERT_EMAIL_TO, and ALERT_EMAIL_FROM must all be set",
        )

    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": from_email},
        "subject": f"Your Top 10 Fresh Jobs This Week — {date.today().isoformat()}",
        "content": [{"type": "text/html", "value": _build_top10_html(jobs)}],
    }
    response = await client.post(
        SENDGRID_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    response.raise_for_status()


@app.get("/version")
async def version() -> dict:
    """Reports which code commit Render actually has running - use this to
    confirm a deploy really picked up the latest push, without relying on
    interpreting the Render dashboard UI."""
    return {
        "commit": os.environ.get("RENDER_GIT_COMMIT", "unknown - not running on Render or var unset"),
        "branch": os.environ.get("RENDER_GIT_BRANCH", "unknown"),
    }


@app.get("/weekly-top10-preview")
async def weekly_top10_preview() -> dict:
    """Run search (past week only) -> filter -> rank, return the top 10 as JSON only."""
    hits = await _run_platform_scan(tbs="qdr:w")
    candidates = await _filter_scan_results(hits)
    top10 = await _rank_top10(candidates)
    return {"raw_hits": len(hits), "candidates": len(candidates), "top10": top10}


@app.post("/weekly-top10")
async def weekly_top10() -> dict:
    """Run the full weekly pipeline (search -> filter -> rank -> email) and return a summary."""
    hits = await _run_platform_scan(tbs="qdr:w")
    candidates = await _filter_scan_results(hits)
    top10 = await _rank_top10(candidates)

    async with httpx.AsyncClient(timeout=30.0) as client:
        await _send_top10_email(client, top10)

    return {"raw_hits": len(hits), "candidates": len(candidates), "top10": top10, "emailed": True}
