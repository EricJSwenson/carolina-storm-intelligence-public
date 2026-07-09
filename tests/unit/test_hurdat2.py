from storm_eval.ingestion.hurdat2 import (
    _parse_coord, saffir_simpson_category, read_hurdat2,
)
from storm_eval.config import SAMPLES_DIR


def test_coord_hemisphere_signs():
    assert _parse_coord("29.1N") == 29.1
    assert _parse_coord("90.2W") == -90.2
    assert _parse_coord("12.0S") == -12.0


def test_saffir_simpson_thresholds():
    assert saffir_simpson_category(80) == 1
    assert saffir_simpson_category(96) == 3
    assert saffir_simpson_category(137) == 5
    assert saffir_simpson_category(33) == 0
    assert saffir_simpson_category(None) is None


def test_sample_parses_with_landfalls():
    storms = dict((s.storm_id, (s, pts)) for s, pts in
                  read_hurdat2(SAMPLES_DIR / "hurdat2_sample.txt"))
    assert {"AL062018", "AL052019", "AL092011"} <= set(storms)
    s, pts = storms["AL062018"]
    assert s.year == 2018 and s.name == "FLORENCE"
    landfalls = [p for p in pts if p.record_id == "L"]
    assert landfalls and saffir_simpson_category(landfalls[0].max_wind_kt) == 1
