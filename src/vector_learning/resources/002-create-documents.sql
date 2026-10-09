CREATE TABLE IF NOT EXISTS documents (
    id              bigserial PRIMARY KEY,
    document_key    text NOT NULL UNIQUE,
    content         text NOT NULL,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector(384) NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now()
);

-- Upgrade a table created from the earlier schema sketch without dropping its rows.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_key text;
UPDATE documents
SET document_key = 'legacy-' || id::text
WHERE document_key IS NULL;
ALTER TABLE documents ALTER COLUMN document_key SET NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS documents_document_key_uidx ON documents (document_key);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();
