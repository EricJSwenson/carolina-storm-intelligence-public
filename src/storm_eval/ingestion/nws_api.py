"""Pull NWS text products from api.weather.gov.

The Newport/Morehead City forecast office (KMHX) issues Area Forecast
Discussions and Coastal Waters Forecasts for the Carolina coast. These are a
continuously-growing, regionally-specific corpus that complements the Storm
Events narratives. The public NWS API requires no key; production schedules a
poll via ``notebooks/01_bronze_ingest.py``.

Offline, this module reads bundled sample products so the demo needs no network.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from storm_eval.config import SAMPLES_DIR

API_BASE = "https://api.weather.gov"
OFFICE = "MHX"  # Newport / Morehead City, NC


@dataclass
class TextProduct:
    product_id: str
    product_type: str        # AFD (forecast discussion), CWF, etc.
    office: str
    issuance_time: str
    text: str

    @property
    def document(self) -> str:
        return self.text.strip()


def fetch_products(product_type: str = "AFD", office: str = OFFICE,
                   limit: int = 25) -> Iterator[TextProduct]:
    """Fetch recent products live from api.weather.gov (production path)."""
    url = f"{API_BASE}/products/types/{product_type}/locations/{office}"
    req = urllib.request.Request(url, headers={"User-Agent": "storm-eval (contact@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        listing = json.load(resp)
    for item in listing.get("@graph", [])[:limit]:
        with urllib.request.urlopen(  # noqa: S310
            urllib.request.Request(item["@id"], headers={"User-Agent": "storm-eval"}), timeout=30
        ) as r:
            doc = json.load(r)
        yield TextProduct(
            product_id=doc["id"], product_type=product_type, office=office,
            issuance_time=doc["issuanceTime"], text=doc["productText"],
        )


def read_text_products(path: Path | str | None = None) -> Iterator[TextProduct]:
    """Read bundled sample NWS products (offline demo path)."""
    path = Path(path) if path else SAMPLES_DIR / "nws_products_sample.json"
    for d in json.loads(Path(path).read_text(encoding="utf-8")):
        yield TextProduct(**d)


def product_rows(path: Path | str | None = None) -> Iterator[dict]:
    for p in read_text_products(path):
        row = p.__dict__.copy()
        row["document"] = p.document
        yield row
