# Data sources

The platform deliberately combines a large natural-language corpus with
independent structured ground truth, so model answers can be graded against
verifiable facts rather than another model's opinion.

## Corpus (what we retrieve over)

| Source | What it is | Scale | Access |
|---|---|---|---|
| NOAA Storm Events Database | Forecaster-authored episode/event narratives | 1.7M+ events, 1950–present | Yearly bulk CSVs at NCEI |
| NWS text products (api.weather.gov) | Area Forecast Discussions, Coastal Waters Forecasts (office KMHX = Newport/Morehead City, NC) | continuously updated | Public JSON API, no key |

## Structured ground truth (what we grade against)

| Source | What it is | Use |
|---|---|---|
| HURDAT2 (NHC) | Atlantic best-track: 6-hourly position, wind, pressure, status, landfall flags, 1851–present | Saffir-Simpson category, landfall facts |
| NDBC buoys (BFTN7, CLKN7, 41064) | Measured wind / pressure / waves off the Carolina coast | Cross-check measured conditions |

## Why this domain

- **Two independent halves.** The narratives describe storms; HURDAT2 measures
  them. A claim like "Florence made landfall as a Category 2" can be checked
  against the best-track record — most RAG demos have no such oracle.
- **Local relevance.** KMHX and the Carolina buoys tie the system to Morehead
  City, NC.
- **A clear scale-up path.** NEXRAD Level II radar (petabytes, on AWS Open Data)
  and ISD surface observations extend the same medallion design to big data.

## Bundled samples

`data/samples/` contains small, illustrative extracts (three Carolina storms:
Florence 2018, Dorian 2019, Irene 2011) so the whole pipeline runs offline. The
HURDAT2 sample is format-correct and parses with the production parser; the
narratives are original summaries written for this project. Replace these with
the full NOAA downloads for production.
