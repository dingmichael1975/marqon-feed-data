"""Military aircraft enrichment DB.

Builds two JSON files for the v2 sidecar to consume:

  data/mil-aircraft/db.json
      ICAO24 (uppercase hex) → {reg, type_code, type_desc, country}
      Built by:
        1. Download wiedehopf/tar1090-db ranges.json (32 military ranges)
        2. Download every db/{prefix}.js shard (gzip JSON)
        3. Decompress + filter to records whose hex falls in a mil range
        4. Cross-reference type_code with icao_aircraft_types.json for a
           friendly description

  data/mil-aircraft/type_photos.json
      type_code → {wiki_title, photo_url, attribution}
      One photo per unique aircraft TYPE (not per aircraft), pulled from
      Wikipedia's REST API which gives back the page thumbnail.

Cron cadence: monthly (workflow: mil-aircraft-monthly.yml). Military
fleet assignments don't change fast — monthly captures retirements /
new acquisitions without burning runner minutes.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import re
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT       = Path(__file__).resolve().parent.parent
OUT_DIR    = ROOT / "data" / "mil-aircraft"
TAR1090DB  = "https://raw.githubusercontent.com/wiedehopf/tar1090-db/master"
WIKI_API   = "https://en.wikipedia.org/api/rest_v1/page/summary"

DB_PREFIXES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F",
    "00","01","02","03","04","05","06","07","08","09",
    "0A","0B","0C","0D","0E","0F",
    "10","11","12","13","14","15","16","17","18","19",
    "1A","1B","1C","1D","1E","1F",
    "20","21","22","23","24","25","26","27","28","29",
    "2A","2B","2C","2D","2E","2F",
    "30","31","32","33","34","35","36","37","38","39",
    "3A","3B","3C","3D","3E","3F",
    "40","41","42","43","44","45","46","47","48","49",
    "4A","4B","4C","4D","4E","4F",
    "50","51","52","53","54","55","56","57","58","59",
    "5A","5B","5C","5D","5E","5F",
    "60","70","71","72","73","74","75","76","77",
    "78","79","7A","7B","7C","7D","7E","7F",
    "80","85","86","87","88","89",
    "8A","8B","8C","8D","8E","8F",
    "90","9A","9B","9C","9D","9E","9F",
    "A0","A1","A2","A3","A4","A5","A6","A7","A8","A9",
    "AA","AB","AC","AD","AE","AF",
    "B","C","C0","C1","C2","C3","D","E0","E4","E8","EC","F",
]
# Note: tar1090-db sharding is single-char OR 2-char depending on density.
# We use the actual published file list (fetched via GitHub API at runtime).


def fetch_ranges() -> list[tuple[int, int]]:
    """Pull mil hex ranges and return list of (lo, hi) int tuples."""
    r = httpx.get(f"{TAR1090DB}/ranges.json", timeout=30.0)
    r.raise_for_status()
    data = r.json()
    out: list[tuple[int, int]] = []
    for lo, hi in data["military"]:
        out.append((int(lo, 16), int(hi, 16)))
    out.sort()
    return out


def in_mil_range(hex_str: str, ranges: list[tuple[int, int]]) -> bool:
    """Binary-search the sorted ranges for whether hex falls inside any."""
    try:
        n = int(hex_str, 16)
    except ValueError:
        return False
    # Linear is fine for 32 ranges
    for lo, hi in ranges:
        if lo <= n <= hi:
            return True
        if n < lo:
            return False
    return False


def fetch_type_descriptions() -> dict[str, dict[str, Any]]:
    """ICAO type code → {desc, wtc}."""
    r = httpx.get(f"{TAR1090DB}/icao_aircraft_types.json", timeout=30.0)
    r.raise_for_status()
    return r.json()


def list_db_shards() -> list[str]:
    """Ask GitHub API for the actual filename list under db/."""
    r = httpx.get(
        "https://api.github.com/repos/wiedehopf/tar1090-db/contents/db",
        timeout=30.0,
    )
    r.raise_for_status()
    out: list[str] = []
    for entry in r.json():
        if entry["type"] != "file":
            continue
        name = entry["name"]
        if name.endswith(".js"):
            out.append(name)
    return sorted(out)


async def fetch_shard(client: httpx.AsyncClient, name: str) -> dict[str, list]:
    """Download one db/{name}.js, gunzip, parse the JS-wrapped JSON object."""
    r = await client.get(f"{TAR1090DB}/db/{name}", timeout=60.0)
    r.raise_for_status()
    text = gzip.decompress(r.content).decode("utf-8", errors="replace")
    # File contents start directly with the JSON object (no `var = ` wrapper
    # for newer dumps). Try both.
    m = re.search(r"=\s*(\{.*\});?\s*$", text, re.S)
    raw = m.group(1) if m else text
    return json.loads(raw)


def _shard_prefix(name: str) -> str:
    """`'39.js'` → `'39'`; `'A4.js'` → `'A4'`."""
    return name[:-3] if name.endswith(".js") else name


def _full_hex(shard_prefix: str, key: str) -> str:
    """Concat shard prefix with shard-internal key, then zero-pad to
    6-char uppercase. tar1090-db stores hex split as `{prefix}{key}`
    where lengths combine to 6 hex chars (e.g. shard `39` has 4-char
    keys, shard `A4` has 4-char keys, shard `1` has 5-char keys).
    Returns canonical ICAO24 form."""
    full = (shard_prefix + key).upper()
    if len(full) < 6:
        full = full.rjust(6, "0")
    return full


async def fetch_all_shards(shard_names: list[str]) -> dict[str, list]:
    """Pull every shard in parallel, merge into one big {hex: [...]} dict.

    Each shard `{prefix}.js` stores keys WITHOUT the prefix; we splice
    the prefix back so the merged map's keys are full 6-char ICAO24.
    Without this step, the keys end up as 4- or 5-char fragments that
    can never match an OpenSky icao24 (always 6-char hex). That bug
    caused 0 % enrichment hit rate on the May 11 run.

    tar1090-db's /db/ also contains support shards (regdb_*, type stats)
    that decode to lists / strings rather than {hex: rec}. Those are
    skipped — only top-level dicts contribute to the merged DB."""
    out: dict[str, list] = {}
    skipped_nondict: list[str] = []
    async with httpx.AsyncClient() as client:
        # Concurrency cap — 16 simultaneous is plenty for GitHub raw.
        sem = asyncio.Semaphore(16)

        async def bounded(name: str) -> tuple[str, Any]:
            async with sem:
                try:
                    rec = await fetch_shard(client, name)
                except Exception as exc:
                    print(f"  [warn] shard {name} failed: {exc}")
                    return name, {}
                return name, rec

        results = await asyncio.gather(*(bounded(n) for n in shard_names))
        for name, d in results:
            if not isinstance(d, dict):
                skipped_nondict.append(f"{name}({type(d).__name__})")
                continue
            prefix = _shard_prefix(name)
            for key, rec in d.items():
                out[_full_hex(prefix, key)] = rec
    if skipped_nondict:
        head = ", ".join(skipped_nondict[:10])
        tail = " ..." if len(skipped_nondict) > 10 else ""
        print(f"  [info] skipped {len(skipped_nondict)} non-dict shards: {head}{tail}")
    return out


# ── Wikipedia photo lookup ──────────────────────────────────────────

# Aircraft TYPE CODE → Wikipedia article title.
#
# Mictronics' type codes are ICAO codes (F16, B52H, C130, etc.).
# Wikipedia article titles don't match these directly — we hand-curate
# a mapping for the common military types we expect to encounter.
# Anything not in here falls back to "{type_code} (aircraft)" which
# misses sometimes, but the JSON only needs to NOT crash on misses.

TYPE_TO_WIKI: dict[str, str] = {
    # US fighters / strike
    "F15":  "McDonnell Douglas F-15 Eagle",
    "F15E": "McDonnell Douglas F-15E Strike Eagle",
    "F16":  "General Dynamics F-16 Fighting Falcon",
    "F18":  "McDonnell Douglas F/A-18 Hornet",
    "F18S": "Boeing F/A-18E/F Super Hornet",
    "F22":  "Lockheed Martin F-22 Raptor",
    "F35":  "Lockheed Martin F-35 Lightning II",
    "A10":  "Fairchild Republic A-10 Thunderbolt II",
    "AV8B": "McDonnell Douglas AV-8B Harrier II",
    "EA18": "Boeing EA-18G Growler",
    # US bombers
    "B1":   "Rockwell B-1 Lancer",
    "B1B":  "Rockwell B-1 Lancer",
    "B2":   "Northrop B-2 Spirit",
    "B52":  "Boeing B-52 Stratofortress",
    "B52H": "Boeing B-52 Stratofortress",
    # US transports / tankers
    "C5":   "Lockheed C-5 Galaxy",
    "C17":  "Boeing C-17 Globemaster III",
    "C130": "Lockheed C-130 Hercules",
    "C130J":"Lockheed Martin C-130J Super Hercules",
    "C141": "Lockheed C-141 Starlifter",
    "C40":  "Boeing C-40 Clipper",
    "K35R": "Boeing KC-135 Stratotanker",
    "KC10": "McDonnell Douglas KC-10 Extender",
    "KC30": "Airbus A330 MRTT",
    "KC46": "Boeing KC-46 Pegasus",
    "K35":  "Boeing KC-135 Stratotanker",
    # US maritime / ISR
    "E3":   "Boeing E-3 Sentry",
    "E3TF": "Boeing E-3 Sentry",
    "E6":   "Boeing E-6 Mercury",
    "E8":   "Northrop Grumman E-8 Joint STARS",
    "P3":   "Lockheed P-3 Orion",
    "P8":   "Boeing P-8 Poseidon",
    "RC135":"Boeing RC-135",
    "U2":   "Lockheed U-2",
    # US helos
    "H60":  "Sikorsky UH-60 Black Hawk",
    "MH60":  "Sikorsky MH-60 Seahawk",
    "AH64": "Boeing AH-64 Apache",
    "CH47": "Boeing CH-47 Chinook",
    "V22":  "Bell Boeing V-22 Osprey",
    # NATO / European
    "EUFI": "Eurofighter Typhoon",
    "RFAL": "Dassault Rafale",
    "TOR":  "Panavia Tornado",
    "A400": "Airbus A400M Atlas",
    "A330": "Airbus A330 MRTT",
    "M2000":"Dassault Mirage 2000",
    "GR4":  "Panavia Tornado",
    # Russian / Chinese / former Soviet
    "SU24": "Sukhoi Su-24",
    "SU25": "Sukhoi Su-25",
    "SU27": "Sukhoi Su-27",
    "SU30": "Sukhoi Su-30",
    "SU34": "Sukhoi Su-34",
    "SU35": "Sukhoi Su-35",
    "SU57": "Sukhoi Su-57",
    "T154": "Tupolev Tu-154",
    "TU22": "Tupolev Tu-22M",
    "TU95": "Tupolev Tu-95",
    "TU160":"Tupolev Tu-160",
    "MG29": "Mikoyan MiG-29",
    "MG31": "Mikoyan MiG-31",
    "IL76": "Ilyushin Il-76",
    "AN12": "Antonov An-12",
    "AN24": "Antonov An-24",
    "AN72": "Antonov An-72",
    "AN124":"Antonov An-124 Ruslan",
    "Y20":  "Xian Y-20",
    "Y8":   "Shaanxi Y-8",
    # ── Civilian airliners (commonly callsign-tagged into mil feed,
    #    e.g. Air France / Lufthansa / Aeroflot govt charters) ─────
    # Boeing 737 family
    "B731": "Boeing 737",
    "B732": "Boeing 737",
    "B733": "Boeing 737 Classic",
    "B734": "Boeing 737 Classic",
    "B735": "Boeing 737 Classic",
    "B736": "Boeing 737 Next Generation",
    "B737": "Boeing 737 Next Generation",
    "B738": "Boeing 737 Next Generation",
    "B739": "Boeing 737 Next Generation",
    "B37M": "Boeing 737 MAX",
    "B38M": "Boeing 737 MAX",
    "B39M": "Boeing 737 MAX",
    "B3XM": "Boeing 737 MAX",
    # Boeing 747 / 757 / 767 / 777 / 787
    "B741": "Boeing 747",
    "B742": "Boeing 747",
    "B743": "Boeing 747",
    "B744": "Boeing 747-400",
    "B748": "Boeing 747-8",
    "B752": "Boeing 757",
    "B753": "Boeing 757",
    "B762": "Boeing 767",
    "B763": "Boeing 767",
    "B764": "Boeing 767",
    "B772": "Boeing 777",
    "B773": "Boeing 777",
    "B77L": "Boeing 777",
    "B77W": "Boeing 777",
    "B778": "Boeing 777X",
    "B779": "Boeing 777X",
    "B788": "Boeing 787 Dreamliner",
    "B789": "Boeing 787 Dreamliner",
    "B78X": "Boeing 787 Dreamliner",
    # Airbus
    "A306": "Airbus A300",
    "A30B": "Airbus A300",
    "A310": "Airbus A310",
    "A318": "Airbus A318",
    "A319": "Airbus A319",
    "A320": "Airbus A320 family",
    "A321": "Airbus A321",
    "A19N": "Airbus A320neo family",
    "A20N": "Airbus A320neo family",
    "A21N": "Airbus A320neo family",
    "A332": "Airbus A330",
    "A333": "Airbus A330",
    "A338": "Airbus A330neo",
    "A339": "Airbus A330neo",
    "A342": "Airbus A340",
    "A343": "Airbus A340",
    "A345": "Airbus A340",
    "A346": "Airbus A340",
    "A359": "Airbus A350 XWB",
    "A35K": "Airbus A350 XWB",
    "A388": "Airbus A380",
    # Embraer / Bombardier
    "E170": "Embraer E-Jet family",
    "E175": "Embraer E-Jet family",
    "E190": "Embraer E-Jet family",
    "E195": "Embraer E-Jet family",
    "E290": "Embraer E-Jet E2 family",
    "E295": "Embraer E-Jet E2 family",
    "BCS1": "Airbus A220",
    "BCS3": "Airbus A220",
    "CRJ1": "Bombardier CRJ100/200",
    "CRJ2": "Bombardier CRJ100/200",
    "CRJ7": "Bombardier CRJ700 series",
    "CRJ9": "Bombardier CRJ700 series",
    "CRJX": "Bombardier CRJ700 series",
    # ATR / Dash / Saab regional
    "AT42": "ATR 42",
    "AT43": "ATR 42",
    "AT45": "ATR 42",
    "AT72": "ATR 72",
    "AT75": "ATR 72",
    "AT76": "ATR 72",
    "DH8A": "De Havilland Canada Dash 8",
    "DH8B": "De Havilland Canada Dash 8",
    "DH8C": "De Havilland Canada Dash 8",
    "DH8D": "De Havilland Canada Dash 8",
    # Russian / Chinese commercial
    "SU95": "Sukhoi Superjet 100",
    "RRJ":  "Sukhoi Superjet 100",
    "MC21": "Irkut MC-21",
    "C919": "Comac C919",
    "ARJ":  "Comac ARJ21",
    "ARJ2": "Comac ARJ21",
    # Gulfstream / large biz jets
    "G5":   "Gulfstream G550",
    "GLF5": "Gulfstream G550",
    "GLF6": "Gulfstream G650",
    "GA5C": "Gulfstream G500/G600",
    "GLEX": "Bombardier Global Express",
    "GL5T": "Bombardier Global 5000",
    "GL7T": "Bombardier Global 7500",
    "CL30": "Bombardier Challenger 300",
    "CL35": "Bombardier Challenger 350",
    "CL60": "Bombardier Challenger 600 series",
    "FA7X": "Dassault Falcon 7X",
    "FA8X": "Dassault Falcon 8X",
    "F900": "Dassault Falcon 900",
    "F2TH": "Dassault Falcon 2000",
    # Beechcraft / Cessna common
    "BE20": "Beechcraft Super King Air",
    "BE40": "Beechcraft King Air",
    "BE9L": "Beechcraft King Air",
    "B350": "Beechcraft Super King Air",
    "C25A": "Cessna CitationJet",
    "C25B": "Cessna CitationJet",
    "C25C": "Cessna CitationJet",
    "C56X": "Cessna Citation Excel",
    "C68A": "Cessna Citation Latitude",
    "C750": "Cessna Citation X",
    # UAV
    "RQ4":  "Northrop Grumman RQ-4 Global Hawk",
    "MQ9":  "General Atomics MQ-9 Reaper",
    "MQ1":  "General Atomics MQ-1 Predator",
    "MQ4":  "Northrop Grumman MQ-4C Triton",
}


def slugify(title: str) -> str:
    return urllib.parse.quote(title.replace(" ", "_"))


# Wikipedia API requires a User-Agent that identifies the bot AND
# provides a way to contact the operator (URL or email). Plain UA
# strings get 403'd. See: https://meta.wikimedia.org/wiki/User-Agent_policy
WIKI_UA = (
    "marketplus-feed-bot/1.0 "
    "(https://github.com/dingmichael1975/marketplus-feed-data; "
    "dingmichael1975@users.noreply.github.com)"
)


async def fetch_wiki_photo(client: httpx.AsyncClient, title: str) -> dict[str, str] | None:
    """Hit Wikipedia REST API for one article, return thumbnail URL +
    attribution. None on miss (no article / no image)."""
    url = f"{WIKI_API}/{slugify(title)}"
    try:
        r = await client.get(url, timeout=20.0, headers={"User-Agent": WIKI_UA})
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    thumb = (data.get("thumbnail") or {}).get("source")
    orig  = (data.get("originalimage") or {}).get("source")
    if not thumb:
        return None
    return {
        "wiki_title":  data.get("title") or title,
        "wiki_url":    data.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "photo_url":   orig or thumb,           # prefer original; fall back
        "thumb_url":   thumb,
        "attribution": "Wikipedia (CC BY-SA)",
    }


async def fetch_all_photos(type_codes: list[str]) -> dict[str, dict]:
    """For each unique type_code we've seen, look up Wikipedia photo."""
    out: dict[str, dict] = {}
    async with httpx.AsyncClient(follow_redirects=True) as client:
        sem = asyncio.Semaphore(8)

        async def one(tc: str):
            wiki_title = TYPE_TO_WIKI.get(tc)
            if not wiki_title:
                return
            async with sem:
                photo = await fetch_wiki_photo(client, wiki_title)
            if photo:
                out[tc] = photo

        await asyncio.gather(*(one(tc) for tc in type_codes))
    return out


# ── Main ────────────────────────────────────────────────────────────

async def main_async() -> int:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    print(f"[mil-db] start @ {now.isoformat()}")

    ranges = fetch_ranges()
    print(f"[mil-db] military hex ranges: {len(ranges)}")

    type_desc = fetch_type_descriptions()
    print(f"[mil-db] ICAO type codes loaded: {len(type_desc)}")

    shards = list_db_shards()
    print(f"[mil-db] db shards to download: {len(shards)}")

    all_records = await fetch_all_shards(shards)
    print(f"[mil-db] total ICAO records: {len(all_records):,}")

    # Filter to military, build output dict
    mil_db: dict[str, dict[str, str]] = {}
    type_counter: dict[str, int] = {}
    for hex_str, rec in all_records.items():
        if not in_mil_range(hex_str, ranges):
            continue
        # rec format: [reg, type_code, flag, type_desc]
        if not isinstance(rec, list) or len(rec) < 3:
            continue
        reg       = rec[0] or ""
        type_code = (rec[1] or "").upper()
        type_str  = ""
        if len(rec) >= 4 and rec[3]:
            type_str = rec[3]
        elif type_code in type_desc:
            type_str = type_desc[type_code].get("desc", "")
        mil_db[hex_str.upper()] = {
            "reg":       reg,
            "type_code": type_code,
            "type_desc": type_str,
        }
        if type_code:
            type_counter[type_code] = type_counter.get(type_code, 0) + 1

    print(f"[mil-db] military aircraft kept: {len(mil_db):,}")
    print(f"[mil-db] unique type codes: {len(type_counter)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "db.json").write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(mil_db),
            "by_hex":       mil_db,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # ── Universal aircraft codes (P6.13e v2 fix) ─────────────────────
    # The mil filter above keeps only ~400 strictly-military aircraft,
    # but OpenSky's callsign-whitelist mil-flights feed pulls in ~150
    # *callsign-tagged* aircraft per snapshot, the majority of which
    # have civilian hex assignments (Air France govt charter, Chinese
    # VIP flights, etc.). Without a universal hex→type lookup, those
    # aircraft show up in tooltip with no Aircraft row, no Registration,
    # no photo — exactly the user-visible bug we're fixing here.
    #
    # Schema: { generated_at, count, by_hex: { "39E692": "BCS3", ... } }
    # Size: ~750k aircraft × ~14 bytes/entry ≈ 10 MB. Each tar1090-db
    # entry is `[reg, type_code, flag, type_desc]`; we keep only
    # type_code in this universal map to stay under the 25 MB GitHub
    # raw distribution sweet-spot. (Backend looks up type_desc from
    # icao_aircraft_types.json by type_code on demand.)
    universal_codes: dict[str, str] = {}
    civil_type_counter: dict[str, int] = {}
    for hex_str, rec in all_records.items():
        if not isinstance(rec, list) or len(rec) < 2:
            continue
        type_code = (rec[1] or "").strip().upper()
        if not type_code:
            continue
        universal_codes[hex_str.upper()] = type_code
        civil_type_counter[type_code] = civil_type_counter.get(type_code, 0) + 1

    (OUT_DIR / "aircraft_codes.json").write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(universal_codes),
            "by_hex":       universal_codes,
        }, ensure_ascii=False, separators=(",", ":")),   # compact: no indent
        encoding="utf-8",
    )
    print(f"[mil-db] aircraft_codes.json: {len(universal_codes):,} entries, "
          f"{len(civil_type_counter)} unique type codes")

    # Photos — pull Wikipedia thumbnails for every type code we've
    # curated AND that actually appears in the data (mil + civil
    # combined). Keeps the photo-set tight; expanding TYPE_TO_WIKI is
    # the lever to raise hit rate, not blasting Wikipedia.
    seen_types: set[str] = set(type_counter) | set(civil_type_counter)
    photo_targets = [tc for tc in seen_types if tc in TYPE_TO_WIKI]
    print(f"[mil-db] fetching Wikipedia photos for {len(photo_targets)} types ...")
    photos = await fetch_all_photos(photo_targets)
    print(f"[mil-db] photos with valid Wikipedia hits: {len(photos)}")

    (OUT_DIR / "type_photos.json").write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(photos),
            "by_type":      photos,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # ICAO type-code → description table; dumped alongside so the
    # backend doesn't have to re-fetch the upstream JSON. Useful for
    # type_desc resolution when looking up an aircraft from the
    # universal codes table (which only carries type_code).
    (OUT_DIR / "type_descriptions.json").write_text(
        json.dumps({
            "generated_at": now.isoformat(),
            "count":        len(type_desc),
            "by_code":      type_desc,
        }, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"[mil-db] done")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
