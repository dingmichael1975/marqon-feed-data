# marketplus-feed-data

Static-snapshot data feed used by **Market+ Terminal v2**. Two scheduled
workflows do all the work; consumers (the v2 local sidecar) just pull
plain JSON from `raw.githubusercontent.com`.

Why this exists: a few upstream feeds are network-blocked or
unstructured in the user's region. We aggregate them once in CI here
and re-host the parsed result so every client gets the same view
without each one re-paying the cost.

---

## What it serves

### Polymarket — hourly

```
data/polymarket/latest.json        — geo-tagged active markets (top by 24 h volume)
data/polymarket/breaking-6h.json   — top 50 by absolute YES-price delta over 6 h
data/polymarket/history/{ts}.json  — rolling 24 h of hourly snapshots (for delta calc)
```

Driven by [`scripts/fetch_polymarket.py`](scripts/fetch_polymarket.py),
scheduled in [`.github/workflows/polymarket-hourly.yml`](.github/workflows/polymarket-hourly.yml).

Geo filtering uses a curated keyword table in the script
(`GEO_TABLE`). Each market's question title is lowercased and the first
keyword hit assigns `lat / lng / region`. Markets with no geographic
interpretation are dropped.

### USNI Fleet Tracker — weekly

```
data/usni-carriers/latest.json     — most recent fleet map, parsed to JSON
data/usni-carriers/YYYY-MM-DD.json — date-stamped historical snapshots
```

Driven by [`scripts/fetch_usni.py`](scripts/fetch_usni.py), scheduled
in [`.github/workflows/usni-weekly.yml`](.github/workflows/usni-weekly.yml).

The USNI News article carries a PNG map with no structured equivalent.
The workflow downloads that PNG and asks **OpenAI gpt-4o-mini vision**
to extract a JSON list of named vessels with estimated lat / lng.

Approx cost: $0.001 / image × 52 weeks ≈ $0.05 / year. Free GitHub
Actions minutes cover the compute side.

---

## Setup

1. Make the repo **public** so consumers can fetch via `raw.githubusercontent.com`.
2. Add **`OPENAI_API_KEY`** as a GitHub Actions Secret
   (Settings → Secrets and variables → Actions → New repository secret).
3. Verify the workflows under the **Actions** tab; trigger each one
   manually once via *"Run workflow"* to seed the first data.

That's it — the cron schedules take over from there.

---

## Consumer endpoints

After setup, these URLs are public and CDN-cached by GitHub:

```
https://raw.githubusercontent.com/<owner>/marketplus-feed-data/main/data/polymarket/latest.json
https://raw.githubusercontent.com/<owner>/marketplus-feed-data/main/data/polymarket/breaking-6h.json
https://raw.githubusercontent.com/<owner>/marketplus-feed-data/main/data/usni-carriers/latest.json
```

The v2 sidecar's `polymarket.py` and `usni_carriers.py` sources fetch
these directly. No auth needed.

---

## Licence

The scripts and workflows in this repo are MIT.

The data they aggregate keeps the original upstream licences:

- **Polymarket** market data — CC BY-style attribution per Polymarket terms.
- **USNI News** Fleet Tracker map — © USNI News, used under
  fair-use research provisions; we publish only the extracted
  numeric positions, not the original image.
