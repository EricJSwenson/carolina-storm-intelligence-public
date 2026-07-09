-- Gold storm-truth table, derived from HURDAT2 (the evaluator's source of truth).
CREATE TABLE IF NOT EXISTS storm_truth (
    storm_id          VARCHAR PRIMARY KEY,
    name              VARCHAR,
    year              INTEGER,
    landfall_category INTEGER,   -- Saffir-Simpson at NC landfall
    peak_wind_kt      INTEGER,
    min_pressure_mb   INTEGER
);
