"""USNI News Fleet Tracker weekly OCR.

Each Tuesday afternoon ET, USNI News publishes a weekly Fleet and Marine
Tracker article with a PNG map showing every deployed US carrier strike
group, amphibious ready group, and supporting unit. There is no
structured API.

This script:

  1. Fetches  https://news.usni.org/category/fleet-tracker
  2. Parses the most recent article link.
  3. Downloads the PNG fleet-tracker map embedded in that article.
  4. Sends the image to OpenAI gpt-4o-mini with a structured-extraction
     prompt.
  5. Writes the parsed JSON to data/usni-carriers/latest.json plus a
     date-stamped copy for the historical track.

Cron cadence: Wednesday 08:00 UTC (set in .github/workflows/usni-weekly.yml).

Cost: gpt-4o-mini vision ≈ $0.001 / image. 52 runs / year ≈ $0.05.

The OPENAI_API_KEY GitHub Secret is required.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "data" / "usni-carriers"
INDEX_URL  = "https://news.usni.org/category/fleet-tracker"
UA         = "marketplus-feed-bot/1.0 (research)"

SYSTEM_PROMPT = """\
You read US Navy fleet-tracker map images and return structured JSON.
The map shows the approximate positions of US Navy carrier strike groups
(CVN), amphibious ready groups (LHA/LHD), and any other named warships,
each labeled with the vessel name and hull number near a position marker.
Some maps also flag escort vessels (cruisers / destroyers) by name.

Return STRICTLY valid JSON, no markdown fences, no prose:

{
  "as_of": "YYYY-MM-DD",
  "vessels": [
    {
      "name":   "USS Gerald R. Ford (CVN-78)",
      "type":   "carrier",            // carrier | amphib | cruiser | destroyer | other
      "region": "Mediterranean Sea",  // e.g. Mediterranean Sea, Persian Gulf, South China Sea
      "lat":    36.5,
      "lng":    18.2,
      "status": "deployed"            // deployed | homeport | transiting | maintenance
    }
  ]
}

Estimate lat/lng from the marker's position on the world map. Be
generous in extraction — include EVERY labeled vessel, not just CVNs.
If you cannot determine a field (e.g. the map shows a vessel without a
status callout), use a sensible default ("deployed" for non-homeport
locations, "homeport" for vessels shown at Norfolk/San Diego/etc.).
"""


def find_latest_article_url() -> str:
    """Pull the fleet-tracker category page and return the first
    article link found."""
    with httpx.Client(timeout=30.0, headers={"User-Agent": UA}) as c:
        r = c.get(INDEX_URL, follow_redirects=True)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # USNI uses standard WP markup; first <h2 class="entry-title"> > <a> is the latest article.
    for h2 in soup.select("h2.entry-title a, h3.entry-title a, h2 a"):
        href = h2.get("href")
        if href and "/fleet-tracker" in href.lower():
            return href
        if href and re.search(r"/usni-news-fleet-and-marine-tracker", href, re.I):
            return href
    raise RuntimeError("Could not locate latest fleet-tracker article link")


def find_map_image_url(article_url: str) -> str:
    """Fetch the article and return the URL of the embedded fleet
    tracker map PNG. USNI articles consistently put the map as the
    first featured image."""
    with httpx.Client(timeout=30.0, headers={"User-Agent": UA}) as c:
        r = c.get(article_url, follow_redirects=True)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # Try OpenGraph image first (most reliable)
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    # Fallback: first <img> in entry-content
    img = soup.select_one(".entry-content img, article img")
    if img and img.get("src"):
        return img["src"]
    raise RuntimeError(f"No map image found in {article_url}")


def download_image(url: str) -> bytes:
    with httpx.Client(timeout=60.0, headers={"User-Agent": UA}) as c:
        r = c.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content


def vision_extract(png_bytes: bytes) -> dict[str, Any]:
    """Send the PNG to OpenAI Vision and parse the JSON response."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract every labeled vessel from this map."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2000,
    )
    text = resp.choices[0].message.content or ""
    return json.loads(text)


def main() -> int:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    print(f"[usni] start @ {now.isoformat()}")

    article = find_latest_article_url()
    print(f"[usni] article: {article}")

    img_url = find_map_image_url(article)
    print(f"[usni] image: {img_url}")

    png = download_image(img_url)
    print(f"[usni] image size: {len(png)} bytes")

    parsed = vision_extract(png)
    vessels = parsed.get("vessels") or []
    print(f"[usni] vessels extracted: {len(vessels)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Attach fetch metadata so consumers can detect staleness
    out = {
        "as_of":        parsed.get("as_of") or now.strftime("%Y-%m-%d"),
        "fetched_at":   now.isoformat(),
        "source_url":   article,
        "image_url":    img_url,
        "count":        len(vessels),
        "vessels":      vessels,
    }

    (OUT_DIR / "latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    date_stem = (parsed.get("as_of") or now.strftime("%Y-%m-%d"))
    (OUT_DIR / f"{date_stem}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[usni] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
