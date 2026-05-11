"""Polymarket hourly snapshot.

Pulls active markets from gamma-api.polymarket.com, filters down to the
subset that has a geographic interpretation (keyword match against a
curated location table), and writes:

  data/polymarket/latest.json       — current snapshot, ready to render
  data/polymarket/breaking-6h.json  — top 50 by |6h price delta|
  data/polymarket/history/{ts}.json — rolling 24 h of hourly snapshots
                                      (used by the breaking-6h computation)

Cron cadence: every hour at :05 (set in .github/workflows/polymarket-hourly.yml).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "data" / "polymarket"
HIST_DIR   = OUT_DIR / "history"
HIST_KEEP  = 26                 # keep ~26 h to bracket the 6 h lookup
GAMMA_URL  = "https://gamma-api.polymarket.com/markets"
FETCH_LIM  = 500                # markets per page
MAX_PAGES  = 20                 # stop after 10k markets max
TIMEOUT_S  = 30.0

# ── Geographic keyword table ────────────────────────────────────────────
#
# Each entry: (lat, lng, label). The Polymarket question title is lower-
# cased and scanned for the first keyword hit. We accept fairly loose
# matches because Polymarket question phrasing is wordy. The keyword
# itself is matched as a substring — keep them specific enough that
# false positives stay rare ("iran" is fine; "us" alone would not be).
#
# Ordered by specificity — more specific keywords first so a market like
# "Will Israel-Hamas war..." matches "hamas" before falling back to
# "israel".

GEO_TABLE: list[tuple[str, float, float, str]] = [
    # Middle East
    ("hamas",          31.5000,  34.4667, "Gaza"),
    ("hezbollah",      33.8547,  35.8623, "Lebanon"),
    ("gaza",           31.5000,  34.4667, "Gaza"),
    ("west bank",      31.9522,  35.2332, "West Bank"),
    ("hormuz",         26.5667,  56.2500, "Strait of Hormuz"),
    ("red sea",        20.2802,  38.5126, "Red Sea"),
    ("houthi",         15.5527,  48.5164, "Yemen"),
    ("yemen",          15.5527,  48.5164, "Yemen"),
    ("syria",          34.8021,  38.9968, "Syria"),
    ("iran",           32.4279,  53.6880, "Iran"),
    ("israel",         31.0461,  34.8516, "Israel"),
    ("saudi",          23.8859,  45.0792, "Saudi Arabia"),
    ("lebanon",        33.8547,  35.8623, "Lebanon"),
    ("qatar",          25.3548,  51.1839, "Qatar"),
    ("uae",            23.4241,  53.8478, "UAE"),

    # East Asia
    ("xi jinping",     39.9042, 116.4074, "Beijing"),
    ("south china sea",15.0000, 115.0000, "South China Sea"),
    ("taiwan",         23.6978, 120.9605, "Taiwan"),
    ("north korea",    40.3399, 127.5101, "North Korea"),
    ("south korea",    35.9078, 127.7669, "South Korea"),
    ("kim jong",       40.3399, 127.5101, "North Korea"),
    ("japan",          36.2048, 138.2529, "Japan"),
    ("china",          35.8617, 104.1954, "China"),

    # Eastern Europe
    ("putin",          55.7558,  37.6173, "Russia"),
    ("zelensky",       50.4501,  30.5234, "Ukraine"),
    ("kyiv",           50.4501,  30.5234, "Kyiv"),
    ("kremlin",        55.7520,  37.6175, "Russia"),
    ("ukraine",        48.3794,  31.1656, "Ukraine"),
    ("russia",         61.5240, 105.3188, "Russia"),
    ("belarus",        53.7098,  27.9534, "Belarus"),

    # Western leaders / EU / NATO
    ("trump",          38.9072, -77.0369, "Washington DC"),
    ("biden",          38.9072, -77.0369, "Washington DC"),
    ("harris",         38.9072, -77.0369, "Washington DC"),
    ("macron",         48.8566,   2.3522, "Paris"),
    ("merz",           52.5200,  13.4050, "Berlin"),
    ("scholz",         52.5200,  13.4050, "Berlin"),
    ("starmer",        51.5074,  -0.1278, "London"),
    ("nato",           50.8466,   4.3528, "NATO HQ"),
    ("european union", 50.8466,   4.3528, "EU"),

    # US politics — generic
    ("fed ",           38.9072, -77.0369, "Federal Reserve"),
    ("federal reserve",38.9072, -77.0369, "Federal Reserve"),
    ("powell",         38.9072, -77.0369, "Federal Reserve"),
    ("election",       38.9072, -77.0369, "US Election"),
    ("congress",       38.8898, -77.0091, "US Congress"),
    ("supreme court",  38.8906, -77.0044, "SCOTUS"),

    # Other geopolitical hotspots
    ("modi",           28.6139,  77.2090, "Delhi"),
    ("india",          20.5937,  78.9629, "India"),
    ("pakistan",       30.3753,  69.3451, "Pakistan"),
    ("venezuela",       6.4238, -66.5897, "Venezuela"),
    ("argentina",     -38.4161, -63.6167, "Argentina"),
    ("brazil",        -14.2350, -51.9253, "Brazil"),
    ("mexico",         23.6345,-102.5528, "Mexico"),
    ("turkey",         38.9637,  35.2433, "Turkey"),
    ("erdogan",        38.9637,  35.2433, "Turkey"),
    ("nigeria",         9.0820,   8.6753, "Nigeria"),
    ("ethiopia",        9.1450,  40.4897, "Ethiopia"),
    ("sudan",          12.8628,  30.2176, "Sudan"),
    ("south africa",  -30.5595,  22.9375, "South Africa"),

    # Crypto / global
    ("bitcoin",        37.7749,-122.4194, "Bitcoin (SF proxy)"),
    ("ethereum",       40.7128, -74.0060, "Ethereum (NYC proxy)"),
]


def fetch_markets() -> list[dict[str, Any]]:
    """Pull every active, non-closed Gamma market, paginated."""
    out: list[dict[str, Any]] = []
    with httpx.Client(timeout=TIMEOUT_S, headers={"User-Agent": "marketplus-feed-bot/1.0"}) as client:
        for page in range(MAX_PAGES):
            params = {
                "active":      "true",
                "closed":      "false",
                "archived":    "false",
                "limit":       str(FETCH_LIM),
                "offset":      str(page * FETCH_LIM),
                "order":       "volume24hr",
                "ascending":   "false",
            }
            r = client.get(GAMMA_URL, params=params)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < FETCH_LIM:
                break
    return out


def yes_price(m: dict[str, Any]) -> float | None:
    """Extract the YES outcome price from a Gamma market record."""
    raw = m.get("outcomePrices")
    if isinstance(raw, str):
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            return None
    elif isinstance(raw, list):
        arr = raw
    else:
        return None
    if not isinstance(arr, list) or not arr:
        return None
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return None


def geo_for(question: str) -> tuple[float, float, str] | None:
    """First keyword hit wins. Returns (lat, lng, label) or None."""
    q = question.lower()
    for kw, lat, lng, label in GEO_TABLE:
        if kw in q:
            return lat, lng, label
    return None


def shape_market(m: dict[str, Any]) -> dict[str, Any] | None:
    """Filter + reshape one Gamma market into our wire format.
    Returns None if the market is irrelevant (no geo / no price)."""
    question = (m.get("question") or "").strip()
    if not question:
        return None
    price = yes_price(m)
    if price is None:
        return None
    geo = geo_for(question)
    if geo is None:
        return None
    lat, lng, region = geo

    slug = m.get("slug") or ""
    return {
        "id":         str(m.get("id") or m.get("conditionId") or slug),
        "question":   question,
        "slug":       slug,
        "url":        f"https://polymarket.com/event/{slug}" if slug else "",
        "yes_price":  round(price, 4),
        "volume_24h": float(m.get("volume24hr") or 0.0),
        "volume":     float(m.get("volume") or 0.0),
        "end_date":   m.get("endDate"),
        "lat":        lat,
        "lng":        lng,
        "region":     region,
        "category":   m.get("category") or "",
    }


def load_old_snapshot(target_dt: datetime) -> dict[str, float] | None:
    """Pick the historical snapshot closest to (now - 6 h) and return a
    {market_id: yes_price} dict for delta calc. Returns None if no
    history file is within ±90 min of the target."""
    if not HIST_DIR.exists():
        return None
    files = sorted(HIST_DIR.glob("*.json"))
    if not files:
        return None
    best_path: Path | None = None
    best_diff = 999999.0
    for p in files:
        try:
            ts = datetime.fromisoformat(p.stem.replace("Z", "+00:00"))
        except ValueError:
            continue
        # Filename stems like "2026-05-11T07" parse offset-naive; the
        # caller's target_dt is always UTC-aware. Promote ts to UTC so
        # the subtraction stays consistent (tz mismatch raises TypeError).
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        diff = abs((ts - target_dt).total_seconds())
        if diff < best_diff:
            best_diff = diff
            best_path = p
    if best_path is None or best_diff > 5400:  # 90 min tolerance
        return None
    try:
        snap = json.loads(best_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        m["id"]: m["yes_price"]
        for m in snap.get("markets", [])
        if "id" in m and "yes_price" in m
    }


def prune_history():
    """Keep only the most recent HIST_KEEP files in the history dir."""
    if not HIST_DIR.exists():
        return
    files = sorted(HIST_DIR.glob("*.json"))
    excess = len(files) - HIST_KEEP
    for f in files[:max(0, excess)]:
        try:
            f.unlink()
        except OSError:
            pass


def main():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    print(f"[poly] start @ {now.isoformat()}")

    raw = fetch_markets()
    print(f"[poly] fetched {len(raw)} active markets")

    shaped = [s for s in (shape_market(m) for m in raw) if s is not None]
    shaped.sort(key=lambda m: m["volume_24h"], reverse=True)
    print(f"[poly] geo-tagged {len(shaped)} markets")

    # Latest snapshot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)

    latest = {
        "generated_at": now.isoformat(),
        "count":        len(shaped),
        "markets":      shaped,
    }
    (OUT_DIR / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Save raw snapshot to history (for future delta computations)
    hist_path = HIST_DIR / f"{now.strftime('%Y-%m-%dT%H')}.json"
    hist_path.write_text(
        json.dumps({"markets": [{"id": m["id"], "yes_price": m["yes_price"]}
                                for m in shaped]},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Compute 6h breaking
    target_dt = now.replace(microsecond=0)
    target_dt = target_dt.fromtimestamp(now.timestamp() - 6 * 3600, tz=timezone.utc)
    old = load_old_snapshot(target_dt)
    if old is None:
        print(f"[poly] no 6h-old snapshot yet (need ≥6 h of history)")
        breaking = []
    else:
        rows = []
        for m in shaped:
            prev = old.get(m["id"])
            if prev is None:
                continue
            delta = m["yes_price"] - prev
            if abs(delta) < 0.01:    # ignore <1% noise
                continue
            rows.append({
                **m,
                "yes_price_6h_ago": prev,
                "delta_6h":         round(delta, 4),
            })
        rows.sort(key=lambda r: abs(r["delta_6h"]), reverse=True)
        breaking = rows[:50]
        print(f"[poly] breaking-6h: {len(breaking)} markets")

    (OUT_DIR / "breaking-6h.json").write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(breaking),
            "markets":      breaking,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    prune_history()
    print(f"[poly] done")


if __name__ == "__main__":
    sys.exit(main() or 0)
