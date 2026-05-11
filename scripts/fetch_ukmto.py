"""UKMTO maritime alerts.

UKMTO (UK Maritime Trade Operations) publishes incident reports on
https://www.ukmto.org/recent-incidents for vessels operating in the
Arabian Gulf, Gulf of Oman, Strait of Hormuz, Red Sea, Gulf of Aden,
and Indian Ocean. No API; just an HTML page rendered server-side.

This script:

  1. Fetches the recent-incidents page (httpx + browser headers).
     Cloudflare on UKMTO is lenient compared to USNI, but we still
     send a real Chrome UA + Sec-Fetch envelope.
  2. Parses the incident list out of the HTML.
  3. Geocodes each incident's textual position (e.g. "23NM northeast
     of DOHA, QATAR") to lat/lng via a curated reference table —
     the reference city's coordinates, not the exact NM offset.
     Most UKMTO incidents cluster around 50 well-known maritime
     reference points so a curated lookup beats running a full
     geocoder for v1.
  4. Writes data/ukmto/latest.json.

Cron cadence: every 4 hours (workflow: ukmto-4h.yml).
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ukmto"
URL     = "https://www.ukmto.org/recent-incidents"

# UKMTO sits behind Cloudflare and returns 403 to GitHub Actions Azure
# IPs even with full browser headers. We scrape through Firecrawl's
# stealth proxy instead — requires FIRECRAWL_API_KEY in GH Secrets.
# Free tier: 500 requests/month; 4 h cron = 180/month.
FIRECRAWL_API = "https://api.firecrawl.dev/v2/scrape"


# ── Geographic reference table ──────────────────────────────────────
#
# Most UKMTO incident bodies anchor on a named place ("17NM NE of
# DOHA"). We geocode by matching the FIRST place keyword found in the
# body and plotting at that place's coordinates. Order matters — more
# specific multi-word names go BEFORE single-word ones so "MINA SAQR"
# isn't shadowed by "MINA" alone.

GEO_TABLE: list[tuple[str, float, float]] = [
    # UAE
    ("KHAWR FAKKAN",       25.3389,  56.3553),
    ("KHOR FAKKAN",        25.3389,  56.3553),
    ("RAS AL KHAIMAH",     25.7895,  55.9432),
    ("MINA SAQR",          25.9933,  56.0533),
    ("JEBEL ALI",          25.0118,  55.0617),
    ("FUJAIRAH",           25.1288,  56.3265),
    ("ABU DHABI",          24.4539,  54.3773),
    ("SHARJAH",            25.3463,  55.4209),
    ("DUBAI",              25.2048,  55.2708),
    # Qatar
    ("RAS LAFFAN",         25.9000,  51.5950),
    ("DOHA",               25.2854,  51.5310),
    # Oman
    ("RAS AL HADD",        22.5328,  59.7898),
    ("MUSCAT",             23.5859,  58.4059),
    # Iran
    ("SIRIK",              26.5119,  57.0823),
    ("KISH ISLAND",        26.5167,  53.9667),
    ("BANDAR ABBAS",       27.1865,  56.2808),
    # Iraq
    ("AL BASRAH",          30.5085,  47.7804),
    # Kuwait
    ("MUBARAK AL KABEER",  29.1933,  48.0867),
    # Saudi
    ("RAS TANURA",         26.6489,  50.1591),
    ("JUBAIL",             27.0046,  49.6464),
    # Yemen
    ("AL HUDAYDAH",        14.7978,  42.9545),
    ("AL MUKALLA",         14.5237,  49.1265),
    # Somalia
    ("MOGADISHU",           2.0469,  45.3182),
    ("GARACAD",             6.9700,  49.3500),
    ("MAREEYO",             8.6500,  49.5000),
    ("EYL",                 7.9803,  49.8167),
    # Bahrain
    ("PORT OF BAHRAIN",    26.2042,  50.6105),
    ("BAHRAIN",            26.0667,  50.5577),
    # Regional zones (lower priority — fallback)
    ("STRAIT OF HORMUZ",   26.5667,  56.2500),
    ("STRAITS OF HORMUZ",  26.5667,  56.2500),
    ("STRAITS\nOF HORMUZ", 26.5667,  56.2500),     # word-wrapped in some posts
    ("IRTC",               13.5000,  48.5000),     # Gulf of Aden corridor
    ("RED SEA",            20.0000,  38.0000),
    ("GULF OF ADEN",       12.0000,  47.0000),
    ("GULF OF OMAN",       24.5000,  58.0000),
    ("PERSIAN GULF",       26.5000,  51.5000),
    ("ARABIAN GULF",       26.5000,  51.5000),
    ("ARABIAN SEA",        15.0000,  65.0000),
    ("INDIAN OCEAN",        0.0000,  73.0000),
    # Country fallbacks last
    ("QATAR",              25.3548,  51.1839),
    ("KUWAIT",             29.3759,  47.9774),
    ("OMAN",               21.4735,  55.9754),
    ("IRAN",               32.4279,  53.6880),
    ("YEMEN",              15.5527,  48.5164),
    ("SOMALIA",             5.1521,  46.1996),
    ("SAUDI ARABIA",       23.8859,  45.0792),
    ("UAE",                23.4241,  53.8478),
    ("UNITED ARAB EMIRATES",23.4241,  53.8478),
]


def fetch_markdown() -> str:
    """Pull the recent-incidents page via Firecrawl stealth proxy.

    Direct httpx (even with full browser headers + Sec-Fetch-*) gets
    403'd by UKMTO's Cloudflare WAF when called from GitHub Actions
    Azure IPs. Firecrawl's stealth proxy uses residential IPs +
    challenge solving so we get the rendered markdown reliably.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FIRECRAWL_API_KEY env not set — add it as a repo secret"
            " (Settings → Secrets and variables → Actions).")
    body = {
        "url":            URL,
        "formats":        ["markdown"],
        "onlyMainContent": True,
        "proxy":          "stealth",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    with httpx.Client(timeout=90.0) as c:
        r = c.post(FIRECRAWL_API, json=body, headers=headers)
        r.raise_for_status()
        payload = r.json()
    if not payload.get("success", True):
        raise RuntimeError(f"Firecrawl failure: {payload}")
    data = payload.get("data") or payload
    md = data.get("markdown") or ""
    if not md:
        raise RuntimeError(f"Firecrawl returned no markdown: keys={list(data.keys())}")
    return md


# Firecrawl-rendered markdown for each incident looks like:
#
#   ### Attack UKMTO \#56
#       - 📅 9 May 2026
#       UKMTO WARNING - 056-26 - ATTACK
#       UKMTO has received a report of an incident 23NM northeast of
#       DOHA, QATAR.
#       ...
#
# The `\#` is Firecrawl's markdown escape for the literal `#` in the
# heading. We strip backslashes from the doc first (they're only in
# headings + a couple of escaped punctuation marks; nothing semantic
# is lost) and then run a clean regex against `### Attack UKMTO #56`.

# Firecrawl wraps each card in a markdown list, so the heading shows up
# as "- ### Attack UKMTO #56" (note the leading bullet). The regex
# tolerates any leading whitespace / dashes before the `###`.
_HEADER_RE = re.compile(
    r"^[\s\-]*###\s+(Advisory|Attack|Suspicious Activity|Hijack)\s+UKMTO\s+#?(\d+)",
    re.I | re.M,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{4})\b",
    re.I,
)


def parse_incidents(md: str) -> list[dict[str, Any]]:
    """Split the markdown at incident headers and return one dict per card."""
    # Strip Firecrawl's `\` escapes — they only appear before the
    # heading `#` and the occasional `.` and have no semantic value
    # for our parsing. Doing this once up-front lets the regex be a
    # plain `#?(\d+)`.
    md = md.replace("\\#", "#").replace("\\.", ".").replace("\\-", "-")
    headers = list(_HEADER_RE.finditer(md))
    out: list[dict[str, Any]] = []
    for i, hm in enumerate(headers):
        kind = hm.group(1).title()
        try:
            number = int(hm.group(2))
        except ValueError:
            continue
        body_start = hm.end()
        body_end   = headers[i + 1].start() if i + 1 < len(headers) else len(md)
        chunk = md[body_start:body_end].strip()
        # Date — first date-looking token in the chunk
        date_str = ""
        dm = _DATE_RE.search(chunk)
        if dm:
            date_str = dm.group(1)
        # Body — strip leading list bullets / icon-line / date-line
        lines: list[str] = []
        for ln in chunk.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(("-", "*")) and _DATE_RE.search(ln):
                continue                             # date bullet
            if ln.startswith("![") or ln.startswith("!["):
                continue                             # icon image lines
            lines.append(ln)
        body = "\n".join(lines)[:1200]
        if not body:
            continue
        out.append({
            "number": number,
            "type":   kind,
            "date":   date_str,
            "body":   body,
        })
    # Dedup by number, latest first wins
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for ev in out:
        if ev["number"] in seen:
            continue
        seen.add(ev["number"])
        unique.append(ev)
    return unique


def geocode(body: str) -> tuple[float, float, str] | None:
    """First keyword hit wins. Returns (lat, lng, label) or None."""
    up = body.upper()
    for kw, lat, lng in GEO_TABLE:
        # ".replace('\n', ' ')" handles UKMTO's occasional mid-sentence
        # line wraps inside a city name.
        if kw in up.replace("\n", " "):
            return lat, lng, kw.title()
    return None


def severity_for(kind: str) -> str:
    """Map UKMTO incident type → severity bucket for the colour ramp."""
    k = kind.upper()
    if "HIJACK" in k:               return "critical"
    if k == "ATTACK":               return "high"
    if "SUSPICIOUS" in k:           return "medium"
    if "ADVISORY" in k:             return "low"
    return "medium"


def main() -> int:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    print(f"[ukmto] start @ {now.isoformat()}")

    md = fetch_markdown()
    print(f"[ukmto] markdown fetched: {len(md)} chars")
    incidents = parse_incidents(md)
    print(f"[ukmto] parsed {len(incidents)} incidents")

    # Attach geo + severity
    enriched: list[dict[str, Any]] = []
    for ev in incidents:
        geo = geocode(ev["body"])
        if geo is None:
            # Skip placeholder cards with no recognisable location
            continue
        lat, lng, region = geo
        enriched.append({
            "number":   ev["number"],
            "type":     ev["type"],
            "severity": severity_for(ev["type"]),
            "date":     ev["date"],
            "region":   region,
            "lat":      lat,
            "lng":      lng,
            "body":     ev["body"],
        })

    print(f"[ukmto] geo-tagged {len(enriched)} incidents")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated_at": now.isoformat(),
        "count":        len(enriched),
        "source_url":   URL,
        "incidents":    enriched,
    }
    (OUT_DIR / "latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[ukmto] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
