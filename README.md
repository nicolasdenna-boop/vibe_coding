# vibe_coding_test

## Job alert scanning (`/scan-and-alert`)

Searches LinkedIn, Indeed, Glassdoor, Monster, Bayt, Naukri, JobStreet,
Wellfound, and BuiltIn (via SerpAPI) for senior Textile, Sales & Commercial,
Board/NED, and Fashion-Tech-commercial roles, filters the results with
Gemini, and emails a digest via SendGrid. Requires `SERPAPI_KEY`,
`GEMINI_API_KEY`, `SENDGRID_API_KEY`, `ALERT_EMAIL_TO`, and
`ALERT_EMAIL_FROM` (see `.env.example`).

- `GET /scan-preview` — runs the search + filter pipeline and returns the
  matches as JSON, without sending an email. Use this to test.
- `POST /scan-and-alert` — runs the full pipeline and emails the digest.

This app doesn't run its own scheduler (a FastAPI process isn't a reliable
long-running cron host). Instead, point your hosting platform's scheduler at
`POST /scan-and-alert` on whatever cadence you want (e.g. daily or weekly):

- **Render**: add a [Cron Job](https://render.com/docs/cronjobs) that runs
  `curl -X POST https://<your-service>.onrender.com/scan-and-alert`
- **Railway**: use a [Cron Schedule](https://docs.railway.app/reference/cron-jobs)
  on a small job that curls the same endpoint
- **GitHub Actions**: a workflow with a `schedule:` trigger that curls the
  endpoint
- **Google Cloud Scheduler**: an HTTP job targeting the endpoint on a cron
  expression

## Weekly Top 10 Fresh Jobs (`/weekly-top10`)

A more selective sibling of `/scan-and-alert`: same platforms and queries,
but restricted to postings from the past week (SerpAPI's `tbs=qdr:w` time
filter) and narrowed down by a second Gemini ranking pass to exactly the 10
most relevant, senior roles overall across Textile, Sales & Commercial,
Board/NED, and Fashion Tech. The email is short on purpose - just a numbered
list of title/company/platform, a link, and a one-line reason for each, so
you click through and apply directly. Uses the same env vars as
`/scan-and-alert` (no new ones required).

- `GET /weekly-top10-preview` — runs search (past week) + filter + rank and
  returns the top 10 as JSON, without emailing anyone. Use this to test.
- `POST /weekly-top10` — runs the full pipeline and emails the digest.

Same rule as above - this app doesn't schedule itself. Point a weekly cron
trigger (e.g. Monday mornings) at `POST /weekly-top10`:

- **Render**: a weekly [Cron Job](https://render.com/docs/cronjobs) running
  `curl -X POST https://<your-service>.onrender.com/weekly-top10`
- **Railway**: a weekly [Cron Schedule](https://docs.railway.app/reference/cron-jobs)
  curling the same endpoint
- **GitHub Actions**: a workflow with a weekly `schedule:` trigger
- **Google Cloud Scheduler**: an HTTP job on a weekly cron expression
