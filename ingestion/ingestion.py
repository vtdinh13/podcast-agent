import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from tqdm import tqdm

from faster_whisper import WhisperModel

from transcription import (
    download_media_file,
    format_time,
    make_queue,
)
import json


def transcribe_episode(audio_path: Path, model: WhisperModel) -> Path:
    """
    Transcribe a single mp3 file and write the output text file next to it.
    """
    transcript_path = audio_path.with_suffix(".txt")

    segments, info = model.transcribe(str(audio_path), vad_filter=True)
    with transcript_path.open("w", encoding="utf-8") as f_out, tqdm(
        total=float(info.duration),
        unit="s",
        desc=f"Transcribing: {audio_path.stem}",
    ) as bar:
        for segment in segments:
            line = f"({format_time(segment.start)}) {segment.text.strip()}"
            f_out.write(line + "\n")
            bar.n = min(segment.end, info.duration)
            bar.refresh()

    return transcript_path


def fetch_episode_metadata(conn: Connection, episode_name: str) -> Dict[str, str]:
    """
    Fetch the RSS metadata needed for storing transcripts.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, ep_name, name_of_podcast
            FROM rss
            WHERE ep_name = %s;
            """,
            (episode_name,),
        )
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"No RSS row found with name: {episode_name}")
    return row


def insert_transcript(
    conn: Connection,
    table_name: str,
    rss_id: int,
    name_of_podcast: str,
    ep_name: str,
    transcript: str,
) -> None:
    """
    Persist the transcript into the specified table.
    """
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {table_name} (
                rss_id, name_of_podcast, ep_name, transcript
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (rss_id) DO NOTHING;
            """,
            (rss_id, name_of_podcast, ep_name, transcript)
        )
        # Ensure that it was inserted
        row = cur.fetchone()
        if row:
            print('Inserted new transcript with id:', row[0])
        else:
            print('Duplicate; skipped.')


def process_queue(
    rss_path: Path,
    media_directory: Path,
    database_url: str,
    transcript_table: str = "transcripts",
    limit: Optional[int] = None,
    model_size_or_path: str = "tiny",
    device: str = "cpu",
    compute_type: str = "int8",
    failed_dir: Optional[Path] = None,
    failed_log: str = "failed_transcripts.json",
) -> None:
    """
    Download, transcribe, and store episodes into Postgres.
    """
    media_directory = Path(media_directory)
    media_directory.mkdir(parents=True, exist_ok=True)

    failed_dir = Path(failed_dir) if failed_dir else media_directory
    failed_dir.mkdir(parents=True, exist_ok=True)
    failed_log_path = failed_dir / failed_log
    failed_entries = []
    if failed_log_path.exists():
        try:
            data = json.loads(failed_log_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                failed_entries = data
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {failed_log_path}; starting a new failure log.")

    def record_failure(episode: str, audio_file: Path, reason: str) -> None:
        failed_entries.append(
            {
                "episode": episode,
                "reason": reason,
                "audio_file": str(audio_file),
                "attempted_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        failed_log_path.write_text(json.dumps(failed_entries, indent=2), encoding="utf-8")


    queue = make_queue(rss_path, media_directory)
    if limit is not None:
        queue = queue[:limit]

    if not queue:
        print("No new episodes to process.")
        return

    model = WhisperModel(
        model_size_or_path=model_size_or_path,
        device=device,
        compute_type=compute_type,
    )

    success_count = 0

    with psycopg.connect(database_url, autocommit=False) as conn:
        for episode in queue:
            episode_name = episode["ep_name"]
            print(f"Processing episode: {episode_name}")
            audio_file_path = media_directory / f"{episode_name}.mp3"

            try:
                download_media_file(media_file=episode, media_directory=media_directory)

                if not audio_file_path.exists():
                    reason = f"Audio file missing at {audio_file_path}"
                    print(reason)
                    record_failure(episode_name, audio_file_path, reason)
                    continue

                try:
                    transcript_path = transcribe_episode(audio_file_path, model)
                except Exception as exc:  # noqa: BLE001
                    reason = f"Transcription error: {exc}"
                    print(reason)
                    record_failure(episode_name, audio_file_path, reason)
                    continue

                if not transcript_path.exists() or transcript_path.stat().st_size == 0:
                    reason = "Transcript missing or empty"
                    print(f"{reason} for: {episode_name}")
                    record_failure(episode_name, audio_file_path, reason)
                    continue

                metadata = fetch_episode_metadata(conn, episode_name)

                with transcript_path.open("r", encoding="utf-8") as f_in:
                    transcript_text = f_in.read()

                insert_transcript(
                    conn=conn,
                    table_name=transcript_table,
                    rss_id=metadata["id"],
                    name_of_podcast=metadata["name_of_podcast"],
                    ep_name=metadata["ep_name"],
                    transcript=transcript_text,
                )
                conn.commit()
                print(f"Podcast successfully ingested: {metadata['name_of_podcast']} : {metadata['ep_name']}")
                if audio_file_path.exists():
                    audio_file_path.unlink()
                    print(f"Deleted: {audio_file_path}")

                success_count += 1
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                reason = f"Unexpected ingestion error: {exc}"
                print(reason)
                record_failure(episode_name, audio_file_path, reason)

    print(f"Number of files successfully ingested: {success_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, transcribe, and store podcast episodes.",
    )
    parser.add_argument(
        "--rss-path",
        type=Path,
        required=True,
        help="Path to the RSS JSON that defines the episodes.",
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        required=True,
        help="Directory where media files and transcripts are stored.",
    )
    parser.add_argument(
        "--database-url",
        default="postgresql://podcast:podcast@localhost:5434/podcast-agent",
        help="Postgres connection string.",
    )
    parser.add_argument(
        "--transcript-table",
        default="transcripts",
        help="Target Postgres table for transcripts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of episodes to process (default: all).",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help="Faster-Whisper model size or local path.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device to use for inference.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Precision to use for Faster-Whisper inference.",
    )
    parser.add_argument(
        "--failed-dir",
        type=Path,
        default=None,
        help="Directory to record failed transcript attempts (default: media-dir/failed_transcripts).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_queue(
        rss_path=args.rss_path,
        media_directory=args.media_dir,
        database_url=args.database_url,
        transcript_table=args.transcript_table,
        limit=args.limit,
        model_size_or_path=args.model,
        device=args.device,
        compute_type=args.compute_type,
        failed_dir=args.failed_dir,
    )


if __name__ == "__main__":
    main()
