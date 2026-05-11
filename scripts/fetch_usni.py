"""USNI News Fleet Tracker weekly OCR.

Each Tuesday afternoon ET, USNI News publishes a weekly Fleet and Marine
Tracker article with a PNG map showing every deployed US carrier strike
group, amphibious ready group, and supporting unit. There is no
structured API.

This script:

  1. Fetches  https://news.usni.org/category/fleet-tracker
  2. Picks the most recent article URL by date pattern.
  3. Downloads the PNG fleet-tracker map embedded in that article.
  4. Sends the image to OpenAI gpt-4o-mini with a structured-extraction
     prompt.
  5. Writes the parsed JSON to:
       data/usni-carriers/latest.json        — newest week
       data/usni-carriers/{date}.json        — date-stamped archive
       data/usni-carriers/recent.json        — last 12 weeks merged,
                                               with per-vessel track
                                               polylines for the layer

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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from openai import OpenAI

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "data" / "usni-carriers"
INDEX_URL  = "https://news.usni.org/category/fleet-tracker"
# Pattern for fleet-tracker article URLs:
#   https://news.usni.org/2026/05/04/usni-news-fleet-and-marine-tracker-may-4-2026
ARTICLE_RE = re.compile(
    r"https?://news\.usni\.org/(\d{4})/(\d{2})/(\d{2})/"
    r"usni-news-fleet-and-marine-tracker[^\"'\s]*",
    re.I,
)
HIST_KEEP  = 12        # keep ~12 weeks of dated snapshots for path tracking
UA         = "Mozilla/5.0 (compatible; marketplus-feed-bot/1.0; +research)"

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
    """Pull the fleet-tracker category page and scan ALL anchor hrefs
    for the canonical Fleet Tracker article URL pattern. Returns the
    URL with the newest YYYY/MM/DD prefix."""
    with httpx.Client(timeout=30.0, headers={"User-Agent": UA}) as c:
        r = c.get(INDEX_URL, follow_redirects=True)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Collect every anchor href matching the article URL pattern, dedup.
    seen: set[str] = set()
    matches: list[tuple[str, str]] = []   # (yyyy-mm-dd, url)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = ARTICLE_RE.match(href)
        if not m:
            continue
        # Normalise: strip query string + fragment, drop trailing slash.
        clean = href.split("?")[0].split("#")[0].rstrip("/")
        if clean in seen:
            continue
        seen.add(clean)
        date_key = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        matches.append((date_key, clean))

    if not matches:
        # Dump a short sample so the failing run gives us a clue.
        sample = r.text[:2000]
        raise RuntimeError(
            "Could not locate latest fleet-tracker article link. "
            f"Found {len(soup.find_all('a', href=True))} anchors but "
            f"none matched ARTICLE_RE. First 2 KB of body:\n{sample}"
        )

    matches.sort(reverse=True)             # newest YYYY-MM-DD first
    return matches[0][1]


def find_map_image_url(article_url: str) -> str:
    """Fetch the article and return the URL of the embedded fleet
    tracker map PNG. USNI articles consistently put the map as the
    first featured image."""
    with httpx.Client(timeout=30.0, headers={"User-Agent": UA}) as c:
        r = c.get(article_url, follow_redirects=True)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        return og["content"]
    img = soup.select_one(".entry-content img, article img, .post-content img")
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


def build_recent_index() -> dict[str, Any]:
    """Merge the last HIST_KEEP dated snapshots into a single payload
    with per-vessel track polylines. Vessel identity is the `name`
    field (trimmed); track points are sorted oldest → newest so
    consumers can draw a polyline with `track[-1]` as the current.

    Layout:
      {
        "generated_at": ...,
        "weeks":   ["2026-05-11", "2026-05-04", ...],   (newest first)
        "vessels": [
          { "name": "USS ...", "type": "carrier",
            "current": {"lat":..., "lng":..., "region":..., "status":...},
            "track":   [{"as_of":..., "lat":..., "lng":...}, ...]  }, ...
        ]
      }
    """
    snapshots: list[tuple[str, dict[str, Any]]] = []
    for p in sorted(OUT_DIR.glob("*.json")):
        if p.name in {"latest.json", "recent.json"}:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        date_key = data.get("as_of") or p.stem
        snapshots.append((date_key, data))

    snapshots.sort(key=lambda kv: kv[0], reverse=True)   # newest first
    keep = snapshots[:HIST_KEEP]
    if not keep:
        return {"generated_at": "", "weeks": [], "vessels": []}

    # Group track points by vessel name across snapshots.
    by_name: dict[str, dict[str, Any]] = {}
    for date_key, snap in reversed(keep):                # oldest first
        for v in snap.get("vessels") or []:
            name = (v.get("name") or "").strip()
            if not name:
                continue
            try:
                lat = float(v["lat"])
                lng = float(v["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            slot = by_name.setdefault(name, {
                "name":  name,
                "type":  v.get("type") or "other",
                "track": [],
                "current": None,
            })
            point = {"as_of": date_key, "lat": lat, "lng": lng}
            slot["track"].append(point)
            slot["current"] = {
                "lat":    lat,
                "lng":    lng,
                "region": v.get("region") or "",
                "status": v.get("status") or "deployed",
            }
            # type may shift week-to-week if labeller is sloppy; keep
            # the most recent non-empty value
            if v.get("type"):
                slot["type"] = v["type"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weeks":        [d for d, _ in keep],
        "vessels":      list(by_name.values()),
    }


def prune_history():
    """Keep only the most recent HIST_KEEP dated archives + latest.json
    + recent.json. Older weekly archives get deleted so the repo doesn't
    grow unboundedly."""
    dated = []
    for p in OUT_DIR.glob("*.json"):
        if p.name in {"latest.json", "recent.json"}:
            continue
        dated.append(p)
    dated.sort(key=lambda p: p.name)
    excess = len(dated) - HIST_KEEP
    for p in dated[:max(0, excess)]:
        try:
            p.unlink()
        except OSError:
            pass


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

    date_stem = out["as_of"]
    (OUT_DIR / f"{date_stem}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # Rebuild the rolling 12-week recent.json (used by the layer for
    # per-vessel track polylines).
    recent = build_recent_index()
    (OUT_DIR / "recent.json").write_text(
        json.dumps(recent, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[usni] recent.json: {len(recent['weeks'])} weeks, "
          f"{len(recent['vessels'])} unique vessels")

    prune_history()
    print(f"[usni] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
