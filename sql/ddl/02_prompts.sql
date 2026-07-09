-- Prompt/template registry so every response is traceable to its template.
CREATE TABLE IF NOT EXISTS prompts (
    template_name VARCHAR,
    version       VARCHAR,
    body          VARCHAR,
    created_ts    TIMESTAMP
);
