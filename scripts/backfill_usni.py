"""One-shot script to OCR the most recent N weeks of USNI Fleet
Tracker articles and seed data/usni-carriers/{date}.json archives.

Runs as a workflow_dispatch step (.github/workflows/usni-backfill.yml).
After committing the archives, recent.json's vessel tracks immediately
have N points each, so the Qt client's polylines show real carrier
movement without waiting weeks for the weekly cron to accumulate them
naturally.

Usage:
    python scripts/backfill_usni.py [N]      # default N = 12

Cost: gpt-4o-mini vision ≈ $0.001 / image. N=12 ≈ $0.012.

Reuses every helper in fetch_usni.py so we never drift from the
weekly-run JSON schema.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Same-folder import.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_usni as fu  # noqa: E402


def _collect_rss_entries(n_target: int) -> list[tuple[str, str]]:
    """Return [(article_url, image_url), ...] for the newest N entries
    in USNI's fleet-tracker RSS feed, sorted newest first."""
    headers = dict(fu.BROWSER_HEADERS)
    headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; feedparser/6.0; "
        "+https://github.com/kurtmckee/feedparser)"
    )
    headers["Accept"] = (
        "application/rss+xml, application/xml;q=0.9, */*;q=0.5"
    )

    with httpx.Client(timeout=30.0, headers=headers) as c:
        r = c.get(fu.RSS_URL, follow_redirects=True)
        r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {
        "content": "http://purl.org/rss/1.0/modules/content/",
        "media":   "http://search.yahoo.com/mrss/",
    }
    items = root.findall("channel/item")
    print(f"[backfill] RSS items available: {len(items)}")

    entries: list[tuple[str, str]] = []
    # Over-fetch a bit in case some items have no image
    for it in items[:n_target * 2]:
        article_url = (it.findtext("link") or "").strip()
        if not article_url:
            continue
        img_url: str | None = None
        enc = it.find("enclosure")
        if enc is not None and enc.get("url"):
            img_url = enc.get("url")
        if not img_url:
            mc = it.find("media:content", ns)
            if mc is not None and mc.get("url"):
                img_url = mc.get("url")
        if not img_url:
            mt = it.find("media:thumbnail", ns)
            if mt is not None and mt.get("url"):
                img_url = mt.get("url")
        if not img_url:
            content = it.findtext("content:encoded", default="",
                                    namespaces=ns)
            m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']',
                          content, re.I)
            if m:
                img_url = m.group(1)
        if img_url:
            entries.append((article_url, img_url))

    entries.sort(
        key=lambda kv: fu._date_from_article_url(kv[0]) or "",
        reverse=True,
    )
    return entries[:n_target]


def main() -> int:
    n_weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"[backfill] target: last {n_weeks} weeks")

    entries = _collect_rss_entries(n_weeks)
    print(f"[backfill] usable RSS entries: {len(entries)}")

    fu.OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0)

    new_count = 0
    skip_count = 0
    fail_count = 0

    for article_url, img_url in entries:
        date_key = fu._date_from_article_url(article_url)
        if not date_key:
            print(f"[backfill] FAIL: cannot parse date from {article_url}")
            fail_count += 1
            continue
        out_path = fu.OUT_DIR / f"{date_key}.json"
        if out_path.exists():
            print(f"[backfill] SKIP: {date_key} already cached")
            skip_count += 1
            continue

        print(f"[backfill] OCR {date_key} <- {article_url}")
        try:
            png = fu.download_image(img_url)
            parsed = fu.vision_extract(png)
            vessels = parsed.get("vessels") or []
            out = {
                "as_of":      date_key,
                "fetched_at": now.isoformat(),
                "source_url": article_url,
                "image_url":  img_url,
                "count":      len(vessels),
                "vessels":    vessels,
            }
            out_path.write_text(
                json.dumps(out, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            print(f"[backfill]   -> {len(vessels)} vessels written")
            new_count += 1
        except Exception as exc:
            print(f"[backfill] FAIL {date_key}: {exc}")
            fail_count += 1

    print(f"[backfill] summary: new={new_count} "
          f"skipped={skip_count} failed={fail_count}")

    # Rebuild recent.json from every archive on disk.
    print("[backfill] rebuilding recent.json ...")
    recent = fu.build_recent_index()
    (fu.OUT_DIR / "recent.json").write_text(
        json.dumps(recent, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[backfill] recent.json: {len(recent['weeks'])} weeks, "
          f"{len(recent['vessels'])} unique vessels")

    # Rebuild vessel_photos.json (preserve existing + re-fetch any new
    # vessels that surfaced in the older weeks).
    key_to_lookup: dict[str, str] = {}
    for v in recent.get("vessels", []):
        raw = v.get("name") or ""
        norm = fu._normalize_vessel_name(raw)
        if not norm or norm in key_to_lookup:
            continue
        key_to_lookup[norm] = raw or norm
    print(f"[backfill] fetching wiki photos for "
          f"{len(key_to_lookup)} unique vessels ...")
    new_photos = fu.fetch_vessel_photos(key_to_lookup)

    photos_path = fu.OUT_DIR / "vessel_photos.json"
    existing: dict = {}
    if photos_path.exists():
        try:
            existing = json.loads(
                photos_path.read_text(encoding="utf-8")
            ).get("by_name", {}) or {}
        except Exception:
            pass
    merged = dict(existing)
    merged.update(new_photos)
    photos_path.write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(merged),
            "by_name":      merged,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"[backfill] vessel photos: {len(new_photos)} new this run, "
          f"{len(merged)} total")
    print("[backfill] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
