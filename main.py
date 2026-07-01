import asyncio
import json
import os
import re
from html import unescape

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


async def _fetch_all_jobs() -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = await asyncio.gather(
            *(fetch_board(client, token) for token in GREENHOUSE_BOARDS),
            return_exceptions=True,
        )

    jobs: list[dict] = []
    for token, result in zip(GREENHOUSE_BOARDS, results):
        if isinstance(result, Exception):
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch jobs for board '{token}': {result}",
            )
        jobs.extend(result)
    return jobs


@app.get("/jobs")
async def get_jobs() -> dict:
    """Fetch jobs from all configured Greenhouse boards concurrently and combine them."""
    jobs = await _fetch_all_jobs()
    public_jobs = [{k: v for k, v in job.items() if k != "description"} for job in jobs]
    return {"count": len(public_jobs), "jobs": public_jobs}


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
    jobs = await _fetch_all_jobs()
    scored_jobs = await _score_jobs_with_gemini(jobs, payload.cv, payload.role)
    scored_jobs.sort(key=lambda job: job["score"], reverse=True)
    return {"count": len(scored_jobs), "jobs": scored_jobs}
