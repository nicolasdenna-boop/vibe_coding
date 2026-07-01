import asyncio
import json
import os
import re
from html import unescape
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="Reeds Jobs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GREENHOUSE_BOARDS = [
    "riskified",
    "fireblocks",
    "pagayais",
    "gongio",
    "lightricks",
    "similarweb",
    "melio",
    "wizinc",
    "yotpo",
    "catonetworks",
]
GREENHOUSE_URL = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
GEMINI_BATCH_SIZE = 25
GEMINI_CONCURRENCY = 5
GEMINI_DESCRIPTION_CHARS = 600

ADZUNA_COUNTRY = "gb"
RECRUITERS_FILE = Path(__file__).parent / "recruiters.json"


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


async def fetch_board(client: httpx.AsyncClient, token: str) -> list[dict]:
    """Fetch all jobs for a single Greenhouse board and tag them with the company."""
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
            }
        )
    return jobs


async def fetch_adzuna(client: httpx.AsyncClient, role: str) -> list[dict]:
    """Fetch jobs matching the target role from the Adzuna API."""
    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        return []

    response = await client.get(
        f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1",
        params={
            "app_id": app_id,
            "app_key": app_key,
            "what": role,
            "results_per_page": 50,
            "content-type": "application/json",
        },
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("results", []):
        company = job.get("company") or {}
        location = job.get("location") or {}
        jobs.append(
            {
                "title": job.get("title"),
                "location": location.get("display_name"),
                "apply_url": job.get("redirect_url"),
                "company": company.get("display_name"),
                "description": _strip_html(job.get("description")),
            }
        )
    return jobs


async def fetch_reed(client: httpx.AsyncClient, role: str) -> list[dict]:
    """Fetch jobs matching the target role from the Reed API."""
    api_key = os.environ.get("REED_API_KEY")
    if not api_key:
        return []

    response = await client.get(
        "https://www.reed.co.uk/api/1.0/search",
        params={"keywords": role},
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
                "company": job.get("employerName"),
                "description": _strip_html(job.get("jobDescription")),
            }
        )
    return jobs


async def fetch_careerjet(client: httpx.AsyncClient, role: str) -> list[dict]:
    """Fetch jobs matching the target role from the Careerjet API."""
    affiliate_id = os.environ.get("CAREERJET_AFFILIATE_ID")
    if not affiliate_id:
        return []

    response = await client.get(
        "http://public-api.careerjet.net/search",
        params={
            "keywords": role,
            "affid": affiliate_id,
            "user_ip": "11.22.33.44",
            "user_agent": "Mozilla/5.0",
            "locale_code": "en_GB",
        },
    )
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        jobs.append(
            {
                "title": job.get("title"),
                "location": job.get("locations"),
                "apply_url": job.get("url"),
                "company": job.get("company"),
                "description": _strip_html(job.get("description")),
            }
        )
    return jobs


async def fetch_jooble(client: httpx.AsyncClient, role: str) -> list[dict]:
    """Fetch jobs matching the target role from the Jooble API."""
    api_key = os.environ.get("JOOBLE_API_KEY")
    if not api_key:
        return []

    response = await client.post(f"https://jooble.org/api/{api_key}", json={"keywords": role})
    response.raise_for_status()
    data = response.json()
    jobs = []
    for job in data.get("jobs", []):
        jobs.append(
            {
                "title": job.get("title"),
                "location": job.get("location"),
                "apply_url": job.get("link"),
                "company": job.get("company"),
                "description": _strip_html(job.get("snippet")),
            }
        )
    return jobs


EXTRA_SOURCES = [
    ("adzuna", fetch_adzuna),
    ("reed", fetch_reed),
    ("careerjet", fetch_careerjet),
    ("jooble", fetch_jooble),
]


async def _fetch_all_jobs(role: str = "") -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        sources = list(GREENHOUSE_BOARDS)
        tasks = [fetch_board(client, token) for token in GREENHOUSE_BOARDS]

        if role:
            for name, fetcher in EXTRA_SOURCES:
                sources.append(name)
                tasks.append(fetcher(client, role))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    jobs: list[dict] = []
    for source, result in zip(sources, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for source '{source}': {result}",
            )
        jobs.extend(result)
    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    jobs = await _fetch_all_jobs()
    public_jobs = [{k: v for k, v in job.items() if k != "description"} for job in jobs]
    return {"count": len(public_jobs), "jobs": public_jobs}


@app.get("/recruiters")
async def get_recruiters() -> list[dict]:
    """Serve the static list of recruiter contacts."""
    with open(RECRUITERS_FILE) as f:
        return json.load(f)


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
        f"JOBS:\n{json.dumps(listing)}\n\n"
        "For every job in JOBS, return its index, a fit score from 0 to 100 "
        "(100 = perfect fit), and a short one-sentence reason for the score that "
        "references the candidate's CV and/or target role."
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
            raise HTTPException(status_code=502, detail=f"Gemini scoring failed: {result}")
        scored_jobs.extend(result)
    return scored_jobs


@app.post("/rank")
async def rank_jobs(payload: RankRequest) -> dict:
    """Rank all fetched jobs by how well they fit the given CV and target role, using Gemini."""
    jobs = await _fetch_all_jobs(payload.role)
    scored_jobs = await _score_jobs_with_gemini(jobs, payload.cv, payload.role)
    scored_jobs.sort(key=lambda job: job["score"], reverse=True)
    return {"count": len(scored_jobs), "jobs": scored_jobs}
