## Ingestion
- Transcription pipeline (`transcription.py`): fetch RSS feed, build the download queue, download audio, transcribe `.mp3` files, and write transcripts to Postgres. Use `--limit` to cap how many new episodes are transcribed per run. Use `--skip-postgres` to leave transcripts on current working directory without inserting into the database.

  ```bash
  python ingest-data/transcription.py \
    --rss-path ingest-data/huberman-rss.json \
    --media-dir ingest-data/media-files/huberman \
    --transcript-table transcripts \
    --limit 20 \ # optional limit per run
    --skip-postgres  # optional: skip writing to Postgres
  ```
- Chunking pipeline (`chunking.py`): turn transcript `.txt` files into chunks of words, embed them with SentenceTransformers (`all-MPNet-base-v2`), and bulk-index into Elasticsearch with duplicate-chunk protection. Includes sliding window chunking, embedding, metadata join, and index creation with a `dense_vector` field.
  ```bash
  python ingest-data/chunking.py \
  --index-name huberman \
  --meta-rss-csv ingest-data/rss-table.csv \
  --media-directory "ingest-data/media-files-sample/huberman/**.txt" \
  --limit 20 # optional limit per run
  ```