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
