# Nicolas Denna — Web CV / Portfolio

An editorial-luxury, hand-coded single-page site. Self-contained (one
`index.html`), works offline, and proves the "heritage craft × technology"
positioning by *being* hand-built.

## Files
- `index.html` — the site (edit this).
- `CV_Nicolas_Denna_2026_Master.pdf` — the downloadable CV (keep it next to
  `index.html` so the "Download CV" button works). Swap for the Hybrid PDF if
  you prefer.
- `media/` — your real assets, already embedded:
  - `interview_shenzhen.mp4` — the Shenzhen TV interview (H.264, plays in every
    browser) + `interview_poster.jpg` (the designed poster card).
  - `IMG_2335 / IMG_2358 / IMG_2348.jpg` — the Silk Road / Uzbekistan photos.
  - `IMG_1919.jpg` — the "La Moda Italiana @ Almaty" poster.

## Check / personalise
- **Add your portrait (high impact):** drop a photo at `media/portrait.jpg`
  (roughly 4:5 portrait). It appears automatically in the "turning point"
  section; until then an elegant "ND" monogram shows in its place. A strong,
  human portrait is the single most memorable element you can add.
- **Interview topic:** the poster card and text say *"heritage and the future of
  textile."* If the interview was about something more specific, edit the
  `<div class="sub">` in `interview_poster` wording and the copy in the
  `RECOGNITION` section. *(The poster is a rendered image — ask me to
  regenerate it if you change its text.)*
- **Silk Road captions:** in the `SILK ROAD` section, edit any `<figcaption>`
  to your real stops (e.g. name the city/mill). If the person in photo II is
  you, consider "At the loom myself — Uzbekistan."
- **Add more photos:** drop files in `media/` and copy a `<figure class="plate">`
  block.
- **Text:** everything is plain HTML — edit in place.

## Publish (pick one, all free, ~2 minutes)
- **Netlify Drop:** drag this folder onto https://app.netlify.com/drop — instant
  URL. Easiest.
- **Vercel:** `vercel` in this folder, or import via the dashboard.
- **GitHub Pages:** push this folder to a repo, then Settings → Pages → deploy
  from branch. Your URL: `https://<user>.github.io/<repo>/`.
- **Custom domain** (optional, most "precious"): buy e.g. `nicolasdenna.com` and
  point it at any of the above.

Then put the link in your LinkedIn "Featured", your email signature, and warm
outreach.
