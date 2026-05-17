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
import hashlib
import json
import os
import re
import sys
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from openai import OpenAI

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "data" / "usni-carriers"
INDEX_URL  = "https://news.usni.org/category/fleet-tracker"
RSS_URL    = "https://news.usni.org/category/fleet-tracker/feed"
# Pattern for fleet-tracker article URLs:
#   https://news.usni.org/2026/05/04/usni-news-fleet-and-marine-tracker-may-4-2026
ARTICLE_RE = re.compile(
    r"https?://news\.usni\.org/(\d{4})/(\d{2})/(\d{2})/"
    r"usni-news-fleet-and-marine-tracker[^\"'\s]*",
    re.I,
)
HIST_KEEP  = 12        # keep ~12 weeks of dated snapshots for path tracking


def _date_from_article_url(url: str) -> str | None:
    """Parse YYYY-MM-DD out of a fleet-tracker article URL — the URL
    path is the canonical source of truth. OpenAI Vision tends to
    hallucinate as_of='2023-10-01' on every image, which caused every
    weekly run to overwrite the same dated archive file and left
    recent.json with only one track point per vessel."""
    m = ARTICLE_RE.match(url)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

# Real-browser headers — USNI's article pages 403 anything that
# self-identifies as a bot or omits the standard Sec-Fetch-* envelope.
# The category listing tolerated a plain UA because it's edge-cached;
# individual article fetches are not, so we always go through here.
BROWSER_HEADERS = {
    "User-Agent":      ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":  "document",
    "Sec-Fetch-Mode":  "navigate",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-User":  "?1",
    "Referer":         "https://news.usni.org/category/fleet-tracker",
}

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
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = "https://news.usni.org/"
    with httpx.Client(timeout=30.0, headers=headers) as c:
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


def fetch_via_rss() -> tuple[str, str]:
    """Pull the category RSS feed and return (article_url, image_url) for
    the most recent fleet-tracker post.

    RSS is the most reliable path on Cloudflare-protected WP sites —
    feed-reader user agents are explicitly whitelisted because USNI
    *wants* Feedly / Inoreader / etc. to redistribute their headlines.
    Article-page HTML and the WP REST API both 403 from GitHub Actions
    Azure IPs (verified 2026-05-11). RSS does not.
    """
    headers = dict(BROWSER_HEADERS)
    # Identify as a feed reader; some Cloudflare WAF rules whitelist
    # User-Agents that mention "feedparser" / "RSS".
    headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; feedparser/6.0; +https://github.com/kurtmckee/feedparser)"
    )
    headers["Accept"] = "application/rss+xml, application/xml;q=0.9, */*;q=0.5"

    with httpx.Client(timeout=30.0, headers=headers) as c:
        r = c.get(RSS_URL, follow_redirects=True)
        r.raise_for_status()

    root = ET.fromstring(r.text)
    ns = {
        "content": "http://purl.org/rss/1.0/modules/content/",
        "media":   "http://search.yahoo.com/mrss/",
    }
    item = root.find("channel/item")
    if item is None:
        raise RuntimeError("RSS feed had no <item> entries")

    article_url = (item.findtext("link") or "").strip()
    if not article_url:
        raise RuntimeError("RSS <item> missing <link>")

    # Pull image URL — try in order: <enclosure>, <media:content>,
    # <media:thumbnail>, then regex over <content:encoded>.
    enc = item.find("enclosure")
    if enc is not None and enc.get("url"):
        return article_url, enc.get("url")

    mc = item.find("media:content", ns)
    if mc is not None and mc.get("url"):
        return article_url, mc.get("url")

    mt = item.find("media:thumbnail", ns)
    if mt is not None and mt.get("url"):
        return article_url, mt.get("url")

    content_html = item.findtext("content:encoded", default="", namespaces=ns)
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content_html, re.I)
    if m:
        return article_url, m.group(1)

    # Final fallback: the description field sometimes has an inline img too.
    description = item.findtext("description") or ""
    m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', description, re.I)
    if m:
        return article_url, m.group(1)

    raise RuntimeError(f"No image found in RSS item for {article_url}")


def _wp_slug_from_url(article_url: str) -> str:
    """Extract the WordPress slug (last URL segment) from an article URL."""
    return article_url.rstrip("/").rsplit("/", 1)[-1]


def find_map_image_url(article_url: str) -> str:
    """Locate the article's featured image (the fleet-tracker map PNG).

    Strategy:
      1. Try WordPress REST API first — USNI exposes
         /wp-json/wp/v2/posts?slug=...&_embed which returns the
         featured-media source_url cleanly, and is rarely bot-gated.
      2. Fall back to HTML scraping with full browser headers if the
         REST endpoint shape changes or 404s.

    USNI's plain article pages return 403 to anything that doesn't
    look like a real Chrome session, so the HTML fallback also uses
    the BROWSER_HEADERS bundle.
    """
    slug = _wp_slug_from_url(article_url)
    api_url = f"https://news.usni.org/wp-json/wp/v2/posts?slug={slug}&_embed"

    # 1) WP REST API
    try:
        with httpx.Client(timeout=30.0, headers=BROWSER_HEADERS) as c:
            r = c.get(api_url, follow_redirects=True)
            r.raise_for_status()
            posts = r.json()
        if posts and isinstance(posts, list):
            post = posts[0]
            embedded = post.get("_embedded") or {}
            media = embedded.get("wp:featuredmedia") or []
            if media:
                src = media[0].get("source_url")
                if src:
                    return src
            # Some posts skip _embed but expose featured_media id
            jp = post.get("jetpack_featured_media_url")
            if jp:
                return jp
    except Exception as exc:
        print(f"[usni] WP REST fallback (API path errored: {exc})")

    # 2) Plain HTML fallback
    headers = dict(BROWSER_HEADERS)
    headers["Referer"] = "https://news.usni.org/category/fleet-tracker"
    with httpx.Client(timeout=30.0, headers=headers) as c:
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
    headers = dict(BROWSER_HEADERS)
    headers["Accept"] = "image/avif,image/webp,image/png,image/*;q=0.8,*/*;q=0.5"
    headers["Referer"] = "https://news.usni.org/"
    with httpx.Client(timeout=60.0, headers=headers) as c:
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
        if p.name in {"latest.json", "recent.json", "vessel_photos.json"}:
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
                "name":      name,
                "photo_key": _photo_key(name),
                "type":      v.get("type") or "other",
                "track":     [],
                "current":   None,
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


_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"

# Wikipedia API requires a User-Agent that identifies the bot AND
# provides a way to contact the operator (URL or email). Plain UA
# strings get 403'd. See: https://meta.wikimedia.org/wiki/User-Agent_policy
_WIKI_UA = (
    "marketplus-feed-bot/1.0 "
    "(https://github.com/dingmichael1975/marketplus-feed-data; "
    "dingmichael1975@users.noreply.github.com)"
)


def _wiki_slug(title: str) -> str:
    return urllib.parse.quote(title.replace(" ", "_"))


def _strip_paren_hull(name: str) -> str:
    """USS Gerald R. Ford (CVN-78) → USS Gerald R. Ford  (Wikipedia
    article titles drop the hull number paren)."""
    return re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()


# OCR sometimes labels strike groups by their group name rather than the
# flagship — e.g. "Abraham Lincoln CSG", "Boxer ARG". Wikipedia has no
# article for the group itself; we want the flagship's article instead.
_GROUP_RE = re.compile(r"^(.*?)\s+(CSG|CVN|ARG|MEU|SAG|ESG)\s*$", re.I)


def _normalize_vessel_name(name: str) -> str:
    """Flagship form used as the storage key in vessel_photos.json:
      "USS Theodore Roosevelt (CVN-71)" → "USS Theodore Roosevelt"
      "Abraham Lincoln CSG"             → "USS Abraham Lincoln"
      "Iwo Jima ARG"                    → "USS Iwo Jima"
    Server's photo_path_for() iterates vessel_photos.json's by_name
    keys and re-hashes each one to compare against the incoming
    /api/monitor/photo/{key} request, so the key here must match the
    string the server runs through vessel_photo_key()."""
    n = re.sub(r"\s*\([^)]+\)\s*$", "", name).strip()
    m = _GROUP_RE.match(n)
    if m:
        return f"USS {m.group(1).strip()}"
    return n


def _photo_key(name: str) -> str:
    """Server-compatible URL key for the Qt client's
    /api/monitor/photo/{key} request. Mirrors the server's
    enrichment._photo_key_for("ves", normalized_name) so the round
    trip — client emits this string, server hashes its by_name keys
    and looks for an equal one — terminates in a hit.

    Format: 'ves-{safe_name}-{md5_6}'  e.g.
        'ves-USS_Theodore_Roosevelt-3f9a02'
    """
    norm = _normalize_vessel_name(name)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", norm).strip("_")[:60]
    digest = hashlib.md5(norm.encode("utf-8")).hexdigest()[:6]
    return f"ves-{safe}-{digest}"


def _fetch_summary(client: httpx.Client, title: str) -> dict | None:
    """Hit Wikipedia REST summary; return JSON or None."""
    try:
        r = client.get(f"{_WIKI_API}/{_wiki_slug(title)}")
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _wiki_search_uss_titles(client: httpx.Client, query: str) -> list[str]:
    """Wikipedia full-text search → list of titles starting with 'USS '."""
    try:
        r = client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  10,
                "format":   "json",
            },
        )
    except Exception:
        return []
    if r.status_code != 200:
        return []
    hits = r.json().get("query", {}).get("search", [])
    return [h["title"] for h in hits if h["title"].startswith("USS ")]


def _resolve_vessel_title(client: httpx.Client, raw: str) -> dict | None:
    """Walk a vessel label down to a Wikipedia article with a thumbnail.

    Order:
      1. Full label with hull number     ("USS Iwo Jima (LHD-7)")
      2. Label with hull stripped         ("USS Iwo Jima")
      3. If it's a strike-group label    ("Abraham Lincoln CSG"):
         a. Direct probe "USS {name}" — Wikipedia often redirects
            unique names (Gerald R. Ford, George H. W. Bush) to the
            actual ship article with a thumbnail.
         b. Full-text search "USS {name} CVN aircraft carrier"
            (or amphibious assault ship for ARG); walk results and
            return the first one whose summary has a thumbnail.
    Returns the summary dict, or None if nothing matched.
    """
    raw_clean = raw.strip()
    if not raw_clean:
        return None

    # 1 + 2: direct candidates
    candidates: list[str] = [raw_clean]
    stripped = _strip_paren_hull(raw_clean)
    if stripped and stripped != raw_clean:
        candidates.append(stripped)
    for t in candidates:
        d = _fetch_summary(client, t)
        if d and d.get("type") != "disambiguation" \
           and (d.get("thumbnail") or {}).get("source"):
            return d

    # 3: strike-group / amphib-group fallback
    m = _GROUP_RE.match(raw_clean)
    if not m:
        return None
    name = m.group(1).strip()
    kind = m.group(2).upper()
    if kind in ("ARG", "MEU", "ESG"):
        query = f"USS {name} amphibious assault ship"
    else:                                    # CSG / CVN / SAG
        query = f"USS {name} aircraft carrier"

    # 3a: bare "USS {name}" — Wikipedia auto-redirects unique names
    d = _fetch_summary(client, f"USS {name}")
    if d and d.get("type") != "disambiguation" \
       and (d.get("thumbnail") or {}).get("source"):
        return d

    # 3b: search → walk
    for t in _wiki_search_uss_titles(client, query):
        d = _fetch_summary(client, t)
        if d and d.get("type") != "disambiguation" \
           and (d.get("thumbnail") or {}).get("source"):
            return d
    return None


def fetch_vessel_photos(key_to_lookup: dict[str, str]) -> dict[str, dict]:
    """For each (storage_key, lookup_name) pair, resolve the Wikipedia
    article and return its thumbnail keyed by storage_key.

    The split between the two names matters: the Qt client looks the
    photo up by `photo_key` (normalised flagship form like 'USS Iwo
    Jima'), but Wikipedia's most specific article needs the hull
    number ('USS Iwo Jima (LHD-7)') to dodge the disambiguation page
    that "USS Iwo Jima" returns. lookup_name is the wiki-friendly raw
    name, storage_key is what the rest of the system reads."""
    out: dict[str, dict] = {}
    headers = {"User-Agent": _WIKI_UA}
    with httpx.Client(timeout=20.0, headers=headers, follow_redirects=True) as c:
        for key, raw in key_to_lookup.items():
            picked = _resolve_vessel_title(c, raw)
            if not picked:
                continue
            thumb = (picked.get("thumbnail") or {}).get("source")
            orig  = (picked.get("originalimage") or {}).get("source")
            out[key] = {
                "wiki_title":  picked.get("title") or key,
                "wiki_url":    picked.get("content_urls", {}).get(
                                   "desktop", {}).get("page", ""),
                "photo_url":   orig or thumb,
                "thumb_url":   thumb,
                "attribution": "Wikipedia (CC BY-SA)",
            }
    return out


def prune_history():
    """Keep only the most recent HIST_KEEP dated archives + latest.json
    + recent.json. Older weekly archives get deleted so the repo doesn't
    grow unboundedly."""
    dated = []
    for p in OUT_DIR.glob("*.json"):
        if p.name in {"latest.json", "recent.json", "vessel_photos.json"}:
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

    # Primary: RSS feed (USNI Cloudflare whitelists feed clients,
    # blocks article HTML + WP REST from GitHub Actions IPs).
    try:
        article, img_url = fetch_via_rss()
        print(f"[usni] via RSS  article: {article}")
        print(f"[usni] via RSS  image:   {img_url}")
    except Exception as exc:
        print(f"[usni] RSS path failed ({exc}), falling back to HTML scrape")
        article = find_latest_article_url()
        print(f"[usni] via HTML article: {article}")
        img_url = find_map_image_url(article)
        print(f"[usni] via HTML image:   {img_url}")

    png = download_image(img_url)
    print(f"[usni] image size: {len(png)} bytes")

    parsed = vision_extract(png)
    vessels = parsed.get("vessels") or []
    print(f"[usni] vessels extracted: {len(vessels)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = {
        "as_of":        _date_from_article_url(article)
                            or parsed.get("as_of")
                            or now.strftime("%Y-%m-%d"),
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

    # Vessel photos (Wikimedia Commons via Wikipedia summary API). One
    # photo per unique vessel name across the 12-week window so the
    # tooltip can render an image even for ships not in the current
    # week's snapshot. Cheap — ~15 HTTP calls per run.
    # vessel_photos.json keys = NORMALISED flagship name (the form the
    # server hashes during photo_path_for()). The lookup name handed
    # to Wikipedia keeps hull numbers so disambiguation works.
    key_to_lookup: dict[str, str] = {}
    for v in recent.get("vessels", []):
        raw = v.get("name") or ""
        norm = _normalize_vessel_name(raw)
        if not norm or norm in key_to_lookup:
            continue
        key_to_lookup[norm] = raw or norm
    print(f"[usni] fetching Wikipedia photos for {len(key_to_lookup)} unique vessels ...")
    vessel_photos = fetch_vessel_photos(key_to_lookup)
    # Merge with the previous vessel_photos.json. Wiki REST occasionally
    # rate-limits or returns no thumbnail on a given run, so blindly
    # overwriting drops every photo we successfully resolved before;
    # users see the popup go from "had a picture" to "blank". Keep the
    # last good photo when this run failed to resolve a vessel.
    photos_path = OUT_DIR / "vessel_photos.json"
    existing: dict = {}
    if photos_path.exists():
        try:
            prior = json.loads(photos_path.read_text(encoding="utf-8"))
            existing = prior.get("by_name", {}) or {}
        except Exception:
            pass
    merged = dict(existing)
    merged.update(vessel_photos)   # successful new lookups win
    photos_path.write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(merged),
            "by_name":      merged,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[usni] vessel photos: {len(vessel_photos)} new this run, "
          f"{len(merged)} total in vessel_photos.json")

    prune_history()
    print(f"[usni] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
