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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ukmto"
URL     = "https://www.ukmto.org/recent-incidents"

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


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


def fetch_html() -> str:
    """Pull the recent-incidents page. Plain httpx + browser headers."""
    with httpx.Client(timeout=30.0, headers=BROWSER_HEADERS) as c:
        r = c.get(URL, follow_redirects=True)
        r.raise_for_status()
        return r.text


# Incident card HTML on ukmto.org looks roughly like:
#   <article>
#     <h3>Attack UKMTO #56</h3>
#     <span>9 May 2026</span>
#     <p>UKMTO has received a report of an incident 23NM northeast of
#        DOHA, QATAR. ...</p>
#   </article>
#
# Markup varies a bit between page rebuilds so we don't rely on
# specific class names — we just walk every element whose text begins
# with one of the known incident type words ("Advisory", "Attack",
# "Suspicious Activity", "Hijack") and then grab the next date-looking
# sibling + the prose body.

_TYPE_RE = re.compile(
    r"^\s*(Advisory|Attack|Suspicious Activity|Hijack)\s+UKMTO\s+#?(\d+)\s*$",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{4})\b",
    re.I,
)


def parse_incidents(html: str) -> list[dict[str, Any]]:
    """Walk the page and return one dict per incident card."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[dict[str, Any]] = []

    # Each incident's <h3> (or similar header tag) contains the title.
    # Try a few heading levels and let the regex filter.
    for header in soup.find_all(re.compile(r"^h[1-6]$", re.I)):
        txt = header.get_text(" ", strip=True)
        m = _TYPE_RE.match(txt)
        if not m:
            continue
        kind = m.group(1).title()
        try:
            number = int(m.group(2))
        except ValueError:
            continue

        # Date — the closest following text/element containing a date.
        date_str = ""
        for sib in header.find_all_next(string=True, limit=12):
            sib_text = str(sib).strip()
            if not sib_text:
                continue
            md = _DATE_RE.search(sib_text)
            if md:
                date_str = md.group(1)
                break

        # Body — the next <p> (or first non-empty text block after the date).
        body = ""
        next_p = header.find_next("p")
        if next_p is not None:
            body = next_p.get_text("\n", strip=True)
        # Some posts only have a <div>; fall back to combined sibling text
        if not body:
            sib_block = []
            for sib in header.find_all_next(string=True, limit=40):
                t = str(sib).strip()
                if not t:
                    continue
                if _TYPE_RE.match(t):
                    break  # next incident starts
                if _DATE_RE.search(t):
                    continue
                sib_block.append(t)
            body = "\n".join(sib_block)

        if not body:
            continue

        out.append({
            "number":  number,
            "type":    kind,
            "date":    date_str,
            "body":    body[:1200],            # keep cards compact
        })

    # Deduplicate by number (the page has one card per incident)
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for ev in out:
        n = ev["number"]
        if n in seen:
            continue
        seen.add(n)
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

    html = fetch_html()
    incidents = parse_incidents(html)
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
