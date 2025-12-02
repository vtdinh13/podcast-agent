
import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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

def create_collection(
    client: QdrantClient,
    collection_name: str,
    distance: str,
    recreate: bool,
    vector_size = 768
) -> None:
    """Ensure the target collection exists with the expected vector settings."""
    distance_map = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID,
    }
    if distance not in distance_map:
        raise ValueError(f"Unsupported distance '{distance}'.")

    if recreate and client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    if client.collection_exists(collection_name=collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=distance_map[distance]
        ),
    )



def stream_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    embedded_batches_iter: Iterable[List[Dict[str, object]]],
) -> int:
    """Upsert embedded chunk batches into Qdrant."""
    total = 0
    for batch_idx, batch in enumerate(
        tqdm(embedded_batches_iter, desc="Streaming to Qdrant"), 1
    ):
        ids = [chunk["chunk_hash"] for chunk in batch]
        vectors = [chunk["embedding"] for chunk in batch]
        payloads = []
        for chunk in batch:
            payload = dict(chunk)
            payload.pop("embedding", None)
            payloads.append(payload)
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
        )
        total += len(batch)
        tqdm.write(
            f"Batch {batch_idx}: upserted {len(batch)} chunks (total {total})"
        )
    return total


def build_qdrant_client(args) -> QdrantClient:
    """Initialize a Qdrant client targeting either local Docker or Qdrant Cloud."""
    if args.target == "local":
        return QdrantClient(
            host=args.local_host,
            port=args.local_port,
            prefer_grpc=args.use_grpc,
        )

    if not args.cloud_url:
        raise ValueError("Missing --cloud-url for Qdrant Cloud target.")
    api_key = args.cloud_api_key or os.getenv("QDRANT_API_KEY")
    if not api_key:
        raise ValueError("Provide --cloud-api-key or set QDRANT_API_KEY.")

    return QdrantClient(
        url=args.cloud_url,
        api_key=api_key,
        prefer_grpc=args.use_grpc,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload transcript embeddings to local or cloud Qdrant."
    )
    parser.add_argument(
        "--parquet-path",
        required=True,
        help="Path to transcripts.parquet.",
    )
    parser.add_argument(
        "--collection-name",
        required=True,
        help="Collection name to upsert into.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Words per chunk window.",
    )
    parser.add_argument(
        "--chunk-step",
        type=int,
        default=450,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=64,
        help="Rows to pull per Parquet batch.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Chunks per embedding call.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max transcript rows to process.",
    )
    parser.add_argument(
        "--distance",
        choices=["cosine", "dot", "euclid"],
        default="cosine",
        help="Vector distance metric.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    parser.add_argument(
        "--target",
        choices=["local", "cloud"],
        default="local",
        help="Destination: dockerized local Qdrant or Qdrant Cloud.",
    )
    parser.add_argument(
        "--local-host",
        default=os.getenv("QDRANT_HOST", "127.0.0.1"),
        help="Host for local Qdrant (ignored for cloud).",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=6333,
        help="Port for local Qdrant (ignored for cloud).",
    )
    parser.add_argument(
        "--cloud-url",
        default=os.getenv("QDRANT_ENDPOINT"),
        help="HTTPS endpoint for Qdrant Cloud.",
    )
    parser.add_argument(
        "--cloud-api-key",
        default=os.getenv("QDRANT_API_KEY"),
        help="API key for Qdrant Cloud.",
    )
    parser.add_argument(
        "--use-grpc",
        action="store_true",
        help="Use gRPC transport where supported.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Entry point for streaming Parquet transcripts into Qdrant."""
    args = parse_args(argv)
    client = build_qdrant_client(args)

    create_collection(
        client=client,
        collection_name=args.collection_name,
        distance=args.distance,
        recreate=args.recreate,
    )

    row_stream = stream_parquet_rows(
        parquet_path=args.parquet_path,
        batch_size=args.parquet_batch_size,
        limit=args.limit,
    )
    chunk_stream = stream_chunks(
        rows=row_stream,
        chunk_size=args.chunk_size,
        chunk_step=args.chunk_step,
    )
    embedded_stream = embed_batches(
        chunks_iter=chunk_stream,
        embedding_batch_size=args.embedding_batch_size,
    )
    total = stream_to_qdrant(
        client=client,
        collection_name=args.collection_name,
        embedded_batches_iter=embedded_stream,
    )
    print(
        f"Uploaded {total} chunks to "
        f"{'local' if args.target == 'local' else 'cloud'} Qdrant collection "
        f"'{args.collection_name}'."
    )


if __name__ == "__main__":
    main()
