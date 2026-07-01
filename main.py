import asyncio
import re
from collections import Counter
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

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "your", "you", "are",
    "have", "will", "our", "a", "an", "in", "on", "at", "of", "to", "is", "as",
    "or", "be", "we", "by", "it", "its", "all", "who", "you'll", "you're",
    "their", "them", "into", "such", "than", "then", "also", "can", "we're",
    "role", "job", "team", "work", "years", "including", "across", "within",
    "new", "one", "more", "about", "using", "other", "any", "not", "but",
    "has", "was", "were", "been", "being", "they", "his", "her", "she", "him",
}


def _tokenize(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z]+", (text or "").lower())
        if len(word) > 2 and word not in _STOPWORDS
    ]


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
    cv_text: str = Field(..., min_length=1, description="Plain text of the candidate's CV")
    target_role: str = Field(..., min_length=1, description="The role the candidate is looking for")
    top_n: int | None = Field(None, gt=0, description="Only return the top N ranked jobs")


def _document_frequency(jobs: list[dict]) -> Counter:
    """Count, across all jobs, how many postings each keyword appears in."""
    doc_freq: Counter = Counter()
    for job in jobs:
        words = set(_tokenize(job.get("title"))) | set(_tokenize(job.get("description")))
        doc_freq.update(words)
    return doc_freq


def _distinctive_keywords(keywords: set[str], doc_freq: Counter, total_jobs: int, max_ratio: float = 0.5) -> set[str]:
    """Drop keywords so generic (present in most postings) that they carry no signal."""
    if not total_jobs:
        return keywords
    return {word for word in keywords if doc_freq.get(word, 0) / total_jobs <= max_ratio}


def _score_job(job: dict, role_keywords: set[str], cv_keywords: set[str], target_role: str) -> tuple[int, str]:
    title_words = set(_tokenize(job.get("title")))
    job_words = title_words | set(_tokenize(job.get("description")))

    # Role fit is judged by the job title only - matching the target role against
    # the free-text description would score unrelated jobs highly just because
    # words like "sales" show up somewhere in almost every posting.
    role_matches = role_keywords & title_words
    cv_matches = cv_keywords & job_words

    role_ratio = len(role_matches) / len(role_keywords) if role_keywords else 0.0
    cv_ratio = len(cv_matches) / len(cv_keywords) if cv_keywords else 0.0

    score = min(100, round(role_ratio * 65 + cv_ratio * 35))

    reason_parts = []
    if role_matches:
        reason_parts.append(
            f"title matches target role '{target_role}' on: {', '.join(sorted(role_matches)[:5])}"
        )
    if cv_matches:
        reason_parts.append(f"overlaps with your CV background in: {', '.join(sorted(cv_matches)[:5])}")
    reason = "; ".join(reason_parts) if reason_parts else "little overlap with your CV or target role"

    return score, reason


@app.post("/jobs/rank")
async def rank_jobs(payload: RankRequest) -> dict:
    """Rank all fetched jobs by how well they fit the given CV and target role."""
    jobs = await _fetch_all_jobs()
    doc_freq = _document_frequency(jobs)

    role_keywords = set(_tokenize(payload.target_role))
    cv_counts = Counter(_tokenize(payload.cv_text))
    cv_keywords = {word for word, _ in cv_counts.most_common(50)} - role_keywords
    cv_keywords = _distinctive_keywords(cv_keywords, doc_freq, len(jobs))

    ranked = []
    for job in jobs:
        score, reason = _score_job(job, role_keywords, cv_keywords, payload.target_role)
        ranked.append(
            {
                **{k: v for k, v in job.items() if k != "description"},
                "score": score,
                "reason": reason,
            }
        )

    ranked.sort(key=lambda job: job["score"], reverse=True)
    if payload.top_n:
        ranked = ranked[: payload.top_n]

    return {"count": len(ranked), "jobs": ranked}
