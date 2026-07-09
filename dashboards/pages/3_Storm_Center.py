"""Storm Center — live NOAA data + the RAG assistant search bar.

Server-side (Streamlit) so it can call government APIs without browser CORS
limits. Every panel degrades gracefully if a feed is down. The live model
estimate is shown beside the official NHC bulletin and clearly labeled as a
model estimate, not a forecast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboards"))

from _bootstrap import ensure_data  # noqa: E402
from storm_eval.live import feeds  # noqa: E402

st.set_page_config(page_title="Storm Center", page_icon="🛰️", layout="wide")
st.title("Storm Center")
st.caption("Live NOAA/NWS conditions and the storm-narrative assistant. "
           "Live model estimates are not official forecasts — follow the National Hurricane Center.")

# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #
DEFAULT = feeds.Location(34.72, -76.73, "Morehead City, NC")

if "loc" not in st.session_state:
    st.session_state.loc = DEFAULT

with st.sidebar:
    st.subheader("Location")
    q = st.text_input("City, ZIP, or 'lat, lon'", "")
    c1, c2 = st.columns(2)
    if c1.button("Set"):
        if "," in q and all(_part.strip().lstrip("-").replace(".", "").isdigit()
                            for _part in q.split(",")[:2]):
            lat, lon = (float(x) for x in q.split(",")[:2])
            st.session_state.loc = feeds.Location(lat, lon, f"{lat:.2f}, {lon:.2f}")
        elif q.strip():
            g = feeds.geocode(q)
            st.session_state.loc = g or st.session_state.loc
    if c2.button("Auto-detect"):
        d = feeds.detect_location()
        if d:
            st.session_state.loc = d
    loc = st.session_state.loc
    st.write(f"📍 **{loc.label}**")
    st.caption(f"{loc.lat:.3f}, {loc.lon:.3f}")

loc = st.session_state.loc


@st.cache_data(ttl=600, show_spinner=False)
def forecast(lat, lon):
    return feeds.get_forecast(lat, lon)


@st.cache_data(ttl=600, show_spinner=False)
def tides(lat, lon):
    return feeds.get_tides(lat, lon)


@st.cache_data(ttl=600, show_spinner=False)
def swell(lat, lon):
    return feeds.get_swell(lat, lon)


@st.cache_data(ttl=600, show_spinner=False)
def active_storms():
    return feeds.get_active_storms()


@st.cache_resource(show_spinner=False)
def _load_model():
    from storm_eval.forecasting.dataset import generate_dataset
    from storm_eval.forecasting.features import build_training_table
    from storm_eval.forecasting.model import load, save, train
    try:
        return load()
    except Exception:  # noqa: BLE001
        X, y, _ = build_training_table(generate_dataset(n_storms=500))
        m = train(X, y)
        save(m)
        return m


@st.cache_resource(show_spinner="Preparing the assistant…")
def _rag():
    ensure_data()
    from storm_eval.rag.pipeline import RAGPipeline
    from storm_eval.rag.vectorstore import LocalVectorStore
    store = LocalVectorStore().load()
    return RAGPipeline(store, model="mock-grounded")


# --------------------------------------------------------------------------- #
# Local conditions
# --------------------------------------------------------------------------- #
st.header(f"Local conditions · {loc.label}")
col1, col2, col3 = st.columns([1.4, 1, 1])

with col1:
    st.subheader("Forecast")
    fc = forecast(loc.lat, loc.lon)
    if fc.status != "ok":
        st.info(f"Forecast {fc.status}")
    else:
        for p in fc.periods:
            st.markdown(f"**{p['name']}** — {p['temp']} · {p['short']} · wind {p['wind']}")

with col2:
    st.subheader("Tides")
    td = tides(loc.lat, loc.lon)
    if td.status != "ok":
        st.info(f"Tides {td.status}")
    else:
        st.caption(f"Station: {td.station}")
        for p in td.predictions:
            st.markdown(f"**{p['type']}** {p['height_ft']} ft · {p['time']}")

with col3:
    st.subheader("Swell / buoy")
    sw = swell(loc.lat, loc.lon)
    if sw.status != "ok":
        st.info(f"Buoy {sw.status}")
    else:
        st.caption(f"Buoy: {sw.station}")
        st.metric("Wave height", f"{sw.wave_height_ft} ft" if sw.wave_height_ft else "—")
        st.metric("Dominant period", f"{sw.dominant_period_s} s" if sw.dominant_period_s else "—")
        st.metric("Water temp", f"{sw.water_temp_f} °F" if sw.water_temp_f else "—")

# --------------------------------------------------------------------------- #
# Active storms + NHC bulletin + model estimate
# --------------------------------------------------------------------------- #
st.divider()
st.header("Active tropical cyclones (NHC)")
storms = active_storms()
if not storms:
    st.success("No active storms reported by the NHC right now (or the feed is unavailable).")
else:
    model = _load_model()
    for s in storms:
        with st.container(border=True):
            a, b = st.columns([1, 1])
            with a:
                st.subheader(f"{s.classification} {s.name}")
                st.markdown(f"**Intensity:** {s.intensity_kt or '—'} kt · "
                            f"**Pressure:** {s.pressure_mb or '—'} mb")
                st.markdown(f"**Position:** {s.lat}, {s.lon} · **Moving:** {s.movement}")
                st.caption("📋 Official NHC bulletin")
                if s.advisory_url:
                    st.link_button("Read NHC public advisory", s.advisory_url)
            with b:
                st.caption("🔬 Model estimate (not official)")
                from storm_eval.live.predict import estimate_landfall

                est = estimate_landfall(s.track, model) if s.track else None
                if est and est.category is not None:
                    st.metric("Estimated landfall category", f"Cat {est.category}")
                    if est.confidence:
                        st.caption(f"model confidence ≈ {est.confidence:.0%}")
                else:
                    st.info("Not enough track history for a model estimate yet.")
                st.caption("⚠️ Model estimate — not an official forecast. Follow the NHC.")

# --------------------------------------------------------------------------- #
# RAG assistant search bar
# --------------------------------------------------------------------------- #
st.divider()
st.header("Ask the storm assistant")
st.caption("Retrieval over NOAA storm narratives, with answers checked against HURDAT2 ground truth.")


query = st.text_input("Your question", "What category was Florence at NC landfall?")
if st.button("Search", type="primary") and query.strip():
    pipe = _rag()
    res = pipe.answer(query)
    st.markdown(f"### {res.answer}")
    from storm_eval.evaluation.metrics import groundedness
    g = groundedness(res.answer, [h.text for h in res.hits])
    st.caption(f"groundedness {g:.2f} · {res.latency_ms:.0f} ms · model {res.model}")
    with st.expander("Sources retrieved"):
        for h in res.hits:
            st.markdown(f"- **{h.doc_id}** (score {h.score:.2f}): {h.text[:240]}…")
