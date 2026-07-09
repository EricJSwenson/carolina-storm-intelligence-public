"""Live NOAA / NWS data feeds (server-side; called from the Streamlit app).

Every function is defensive: on any network/parse error it returns an empty
result with a status string, so a single down feed never breaks the page. These
hit public, no-key government APIs:

  * NWS api.weather.gov            local point forecast (requires User-Agent)
  * NOAA CO-OPS tidesandcurrents   tide predictions (nearest station)
  * NDBC                           buoy wave/swell observations (nearest station)
  * NHC CurrentStorms.json         active tropical cyclones + advisories

NOTE: these are live calls; they cannot be exercised from an offline CI sandbox.
Run the Storm Center page to verify against the live endpoints.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional

UA = {"User-Agent": "carolina-storm-intelligence (contact: ericjoeswenson@gmail.com)"}


def _get(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return r.read()


def _get_json(url: str, timeout: int = 20) -> dict:
    return json.loads(_get(url, timeout))


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #
@dataclass
class Location:
    lat: float
    lon: float
    label: str = ""


def detect_location() -> Optional[Location]:
    """Approximate location from IP (ipapi.co). Returns None on failure."""
    try:
        d = _get_json("https://ipapi.co/json/")
        return Location(float(d["latitude"]), float(d["longitude"]),
                        f"{d.get('city','')}, {d.get('region_code','')}".strip(", "))
    except Exception:  # noqa: BLE001
        return None


def geocode(query: str) -> Optional[Location]:
    """Geocode a city/ZIP via the free OSM Nominatim API."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&limit=1&q={urllib.parse.quote(query)}"
        res = _get_json(url)
        if res:
            return Location(float(res[0]["lat"]), float(res[0]["lon"]), res[0].get("display_name", query))
    except Exception:  # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------- #
# NWS point forecast
# --------------------------------------------------------------------------- #
@dataclass
class Forecast:
    location: str = ""
    periods: List[dict] = field(default_factory=list)   # name, temp, short, wind
    office: str = ""
    status: str = "ok"


def get_forecast(lat: float, lon: float) -> Forecast:
    try:
        meta = _get_json(f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}")["properties"]
        rel = meta.get("relativeLocation", {}).get("properties", {})
        city = f"{rel.get('city','')}, {rel.get('state','')}".strip(", ")
        fc = _get_json(meta["forecast"])["properties"]["periods"]
        periods = [{"name": p["name"], "temp": f"{p['temperature']}°{p['temperatureUnit']}",
                    "short": p["shortForecast"], "wind": f"{p['windSpeed']} {p['windDirection']}",
                    "icon": p.get("icon", "")} for p in fc[:6]]
        return Forecast(city or f"{lat:.2f},{lon:.2f}", periods, meta.get("cwa", ""))
    except Exception as exc:  # noqa: BLE001
        return Forecast(status=f"unavailable: {exc}")


# --------------------------------------------------------------------------- #
# NOAA CO-OPS tides
# --------------------------------------------------------------------------- #
@dataclass
class Tides:
    station: str = ""
    station_id: str = ""
    predictions: List[dict] = field(default_factory=list)   # time, type (H/L), height_ft
    status: str = "ok"


def _nearest_tide_station(lat: float, lon: float) -> Optional[tuple]:
    url = ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
           "?type=tidepredictions")
    stations = _get_json(url).get("stations", [])
    best, bestd = None, 1e18
    for s in stations:
        d = haversine(lat, lon, float(s["lat"]), float(s["lng"]))
        if d < bestd:
            best, bestd = s, d
    return (best["id"], best["name"]) if best else None


def get_tides(lat: float, lon: float) -> Tides:
    try:
        st = _nearest_tide_station(lat, lon)
        if not st:
            return Tides(status="no station found")
        sid, name = st
        url = ("https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
               f"?product=predictions&application=storm-intel&datum=MLLW&time_zone=lst_ldt"
               f"&units=english&interval=hilo&format=json&date=today&station={sid}")
        preds = _get_json(url).get("predictions", [])
        out = [{"time": p["t"], "type": "High" if p["type"] == "H" else "Low",
                "height_ft": round(float(p["v"]), 1)} for p in preds]
        return Tides(name, sid, out)
    except Exception as exc:  # noqa: BLE001
        return Tides(status=f"unavailable: {exc}")


# --------------------------------------------------------------------------- #
# NDBC buoy swell
# --------------------------------------------------------------------------- #
@dataclass
class Swell:
    station: str = ""
    wave_height_ft: Optional[float] = None
    dominant_period_s: Optional[float] = None
    water_temp_f: Optional[float] = None
    status: str = "ok"


def _nearest_buoy(lat: float, lon: float) -> Optional[str]:
    root = ET.fromstring(_get("https://www.ndbc.noaa.gov/activestations.xml"))
    best, bestd = None, 1e18
    for s in root.findall("station"):
        try:
            d = haversine(lat, lon, float(s.get("lat")), float(s.get("lon")))
        except (TypeError, ValueError):
            continue
        if d < bestd:
            best, bestd = s.get("id"), d
    return best


def get_swell(lat: float, lon: float) -> Swell:
    try:
        sid = _nearest_buoy(lat, lon)
        if not sid:
            return Swell(status="no buoy found")
        txt = _get(f"https://www.ndbc.noaa.gov/data/realtime2/{sid.upper()}.txt").decode()
        lines = [ln for ln in txt.splitlines() if ln and not ln.startswith("#")]
        if not lines:
            return Swell(sid, status="no recent data")
        cols = lines[0].split()
        # Standard meteorological file columns: WVHT(8) DPD(9) WTMP(14) (meters/°C)
        def val(i, conv):
            try:
                v = cols[i]
                return None if v in ("MM", "999.0") else conv(float(v))
            except (IndexError, ValueError):
                return None
        wvht_m = val(8, lambda x: x)
        return Swell(
            station=sid,
            wave_height_ft=round(wvht_m * 3.281, 1) if wvht_m is not None else None,
            dominant_period_s=val(9, lambda x: x),
            water_temp_f=val(14, lambda c: round(c * 9 / 5 + 32, 1)),
        )
    except Exception as exc:  # noqa: BLE001
        return Swell(status=f"unavailable: {exc}")


# --------------------------------------------------------------------------- #
# NHC active storms
# --------------------------------------------------------------------------- #
@dataclass
class ActiveStorm:
    id: str
    name: str
    classification: str
    intensity_kt: Optional[int]
    lat: Optional[float]
    lon: Optional[float]
    movement: str = ""
    pressure_mb: Optional[int] = None
    advisory_url: str = ""
    track: List[dict] = field(default_factory=list)   # forecast cone points


def get_active_storms() -> List[ActiveStorm]:
    """Active Atlantic+Pacific cyclones from NHC. Empty list when none/no data."""
    try:
        data = _get_json("https://www.nhc.noaa.gov/CurrentStorms.json")
        out = []
        for s in data.get("activeStorms", []):
            out.append(ActiveStorm(
                id=s.get("id", ""), name=s.get("name", "UNNAMED"),
                classification=s.get("classification", ""),
                intensity_kt=_to_int(s.get("intensity")),
                lat=_to_float(s.get("latitudeNumeric")),
                lon=_to_float(s.get("longitudeNumeric")),
                movement=f"{s.get('movementDir','')} {s.get('movementSpeed','')} kt".strip(),
                pressure_mb=_to_int(s.get("pressure")),
                advisory_url=s.get("publicAdvisory", {}).get("url", "") if isinstance(s.get("publicAdvisory"), dict) else "",
            ))
        return out
    except Exception:  # noqa: BLE001
        return []


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


