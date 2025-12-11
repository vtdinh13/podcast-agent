import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))


@dataclass
class TranscriptRow:
    """Structured representation for a row loaded from the Parquet file."""
    row_id: int
    rss_id: int
    podcast_name: str
    episode_name: str
    transcript: str

@dataclass
class ChunkingConfig:
    embedding_model = SentenceTransformer("all-MPNet-base-v2")
    TIMESTAMP_LINE = re.compile(r"\((\d{2}:\d{2}:\d{2})\)\s*(.+)")


def stream_parquet_rows(
    parquet_path: str, batch_size: int, limit: Optional[int]
) -> Iterator[TranscriptRow]:
    """Stream rows from a parquet file"""
    parquet_file = pq.ParquetFile(parquet_path)
    yielded = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        frame = batch.to_pandas()
        for record in frame.to_dict("records"):
            transcript = record.get("transcript")
            if not isinstance(transcript, str) or not transcript.strip():
                continue
            yield TranscriptRow(
                row_id=int(record.get("id")),
                rss_id=int(record.get("rss_id")),
                podcast_name=str(record.get("name_of_podcast", "")),
                episode_name=str(record.get("ep_name", "")),
                transcript=transcript,
            )
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def parse_transcript_lines(transcript: str, config: Optional[ChunkingConfig]=None) -> List[Dict[str, str]]:
    """Parse a transcript block into timestamped text entries."""
    if not config:
        config = ChunkingConfig()

    entries = []
    for raw_line in transcript.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        match = config.TIMESTAMP_LINE.match(raw_line)
        if not match:
            continue
        timestamp, content = match.groups()
        if content:
            entries.append({"timestamp": timestamp, "content": content})
    return entries

def flatten_entries(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Flatten timestamped sentences into word-level entries retaining timestamps."""
    flattened: List[Dict[str, str]] = []
    for entry in entries:
        timestamp = entry["timestamp"]
        for word in entry["content"].split():
            flattened.append({"word": word, "timestamp": timestamp})
    return flattened


def create_chunks_for_row(
    row: TranscriptRow, chunk_size: int, chunk_step: int
) -> List[Dict[str, object]]:
    """Generate sliding-window word chunks."""
    entries = parse_transcript_lines(row.transcript)
    flattened_words = flatten_entries(entries)
    if not flattened_words:
        return []

    chunks = []
    total = len(flattened_words)
    for start_idx in range(0, total, chunk_step):
        window = flattened_words[start_idx : start_idx + chunk_size]
        if not window:
            break
        words = [words["word"] for words in window]
        times = [words["timestamp"] for words in window]
        chunk_text = " ".join(words)
        chunk_hash = hashlib.md5(
            f"{row.episode_name}-{times[0]}-{chunk_text}".encode("utf-8")
        ).hexdigest()
        chunk_meta = {
            "name_of_podcast": row.podcast_name,
            "rss_id": row.rss_id,
            "episode_name": row.episode_name,
            "source_row_id": row.row_id,
            "start": times[0],
            "end": times[-1],
            "chunk": chunk_text,
            "chunk_hash": chunk_hash,
        }
        chunks.append(chunk_meta)
        if start_idx + chunk_size >= total:
            break
    return chunks

def stream_chunks(
    rows: Iterable[TranscriptRow], chunk_size: int, chunk_step: int
) -> Iterator[Dict[str, object]]:
    """Yield chunks for each transcript row."""
    for row in rows:
        for chunk in create_chunks_for_row(row, chunk_size, chunk_step):
            yield chunk

def create_embeddings(chunks: List[Dict[str, object]], config: Optional[ChunkingConfig]=None) -> List[Dict[str, object]]:
    """Attach sentence-transformer embeddings to each chunk."""

    if not config:
        config = ChunkingConfig()

    texts = [chunk["chunk"] for chunk in chunks]
    vectors = config.embedding_model.encode(
        texts,
        batch_size=min(32, len(texts)),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    embeddings = []
    for chunk, vector in zip(chunks, vectors):
        copy = dict(chunk)
        copy["embedding"] = vector.tolist()
        embeddings.append(copy)
    return embeddings

def embed_batches(
    chunks_iter: Iterable[Dict[str, object]], embedding_batch_size: int
) -> Iterator[List[Dict[str, object]]]:
    """Consume the chunk stream and yield batched embeddings."""
    chunks_list = []
    for chunk in chunks_iter:
        chunks_list.append(chunk)
        if len(chunks_list) >= embedding_batch_size:
            yield create_embeddings(chunks_list)
            chunks_list = []
    if chunks_list:
        yield create_embeddings(chunks_list)
