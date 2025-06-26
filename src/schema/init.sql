CREATE TABLE IF NOT EXISTS translations (
    id SERIAL PRIMARY KEY,
    original TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT '',
    translation TEXT NOT NULL,
    last_used TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_translation UNIQUE (original, context)
);
