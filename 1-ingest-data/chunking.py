from pathlib import Path
import glob
import re

from typing import List, Any, Dict
import pandas as pd
from tqdm import tqdm
import hashlib
import argparse

from elasticsearch import Elasticsearch, helpers, exceptions

# Increase default timeout to reduce ConnectionTimeout errors during bulk indexing.
es = Elasticsearch("http://localhost:9200", request_timeout=120)

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MPNet-base-v2')

def prepare_entries(path: str):
    match_expression = re.compile(r"\((\d{2}:\d{2}:\d{2})\)\s*(.*)")

    entries = []
    
    path = Path(path)
    episode_name = path.stem

    with open(path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            match = match_expression.match(line)
            if not match:
                continue
            entries.append(
                {
                    'episode_name': episode_name,
                    'start': match.group(1),
                    'content': match.group(2)
                }
            )
    return entries

def flatten_list(entries: List[Any]):
    flattened_list = []
    for i in entries:
        start = i['start']
        episode_name = i['episode_name']
        for line in i['content'].split():
            flattened_list.append(
                (line, start, episode_name)
            )
    return flattened_list


def create_chunks(path:str, size:int, step:int):
    chunks = []
    
    entries = prepare_entries(path)
    flattened_list = flatten_list(entries)

    n = len(flattened_list)
    for i in range(0, n, step):
        batch = flattened_list[i:i+size]
       
        if not batch:
            break

        words = [w for (w, ts, _) in batch]
        times = [ts for (_, ts, _) in batch]
        episode_names = [ep for (_, _, ep) in batch]
        chunk = ' '.join(words)
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        start = times[0]
        episode_name = episode_names[0]

        if i + size < len(flattened_list):
            end = flattened_list[i + size][1]
        else:
            end = times[-1]
        
        chunks.append({
            'episode_name': episode_name,
            'start': start,
            'end': end,
            'chunk': chunk,
            'chunk_hash': chunk_hash
        }
    )
        if i + size >= n:
            break

    return chunks
            
def create_embeddings(chunks:List[Dict[str,Any]]):
    chunk_embeddings = []
    for chunk in tqdm(chunks, desc='Embedding chunks:', total=len(chunks)):
        emb = embedding_model.encode(chunk['chunk'], batch_size=32, convert_to_numpy=True)
    
        chunk_embeddings.append({
            **chunk,
            'embedding': emb})
    return chunk_embeddings

def finalize_chunks_with_meta(chunks_with_embeddings:List[Dict[str, Any]], meta_rss_csv:str):
    meta_rss_csv_path = Path(meta_rss_csv)
    rss = pd.read_csv(meta_rss_csv_path)
    complete_chunks = []
    
    for i, j in rss.iterrows():
        for k in chunks_with_embeddings:
            if j['ep_name'] == k['episode_name']:
                complete_chunks.append(
                    {   'name_of_podcast': j['name_of_podcast'],
                        'rss_id': j['id'],
                        **k
                        
                    }
                )
    return complete_chunks

def create_complete_chunks(path:str, size:int, step:int, meta_rss_csv:str):
    meta_rss_csv_path = Path(meta_rss_csv)
    initial_chunks = create_chunks(path, size, step)
    chunks_with_embs = create_embeddings(initial_chunks)
    complete_chunks = finalize_chunks_with_meta(chunks_with_embs, meta_rss_csv_path)
    return complete_chunks

def index(index_name:str, complete_chunks:list):
    index_settings = {
    'mappings': {
        'properties': {
            'name_of_podcast': {'type': 'text'},
            'rss_id': {'type': 'integer'},
            'episode_name': {'type':'text'},
            'start': {'type':'text'},
            'end': {'type': 'text'},
            'chunk': {'type':'text'}, 
            # 'chunk_hash': {'type': 'text'},
            'embedding': {'type': 'dense_vector', 'dims':768, 'index':True, 'similarity': 'cosine'}
        }
    }
}
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=index_settings)
    
    actions = []
    for chunk in tqdm(complete_chunks, total=len(complete_chunks), desc="Preparing index actions:"):
        doc_id = chunk['chunk_hash']
        actions.append(
            {
                "_index": index_name,
                "_id": doc_id,
                "_op_type": "create",
                "_source": chunk,
            }
        )
    if actions:
        try:
            # Apply request timeout via options() to avoid deprecated transport kwargs.
            es_with_timeout = es.options(request_timeout=120)
            success_count, errors = helpers.bulk(
                es_with_timeout,
                actions,
                raise_on_error=False  # let us inspect conflicts explicitly
            )
            conflict_errors = []
            other_errors = []
            for err in errors:
                for _, meta in err.items():
                    status = meta.get('status')
                    if status == 409:
                        conflict_errors.append(err)
                    else:
                        other_errors.append(err)

            if success_count:
                print(f"Successfully indexed {success_count} docs into {index_name}.")
            if conflict_errors:
                print(f"Skipped {len(conflict_errors)} duplicate chunks (409 conflicts).")
            if other_errors:
                print(f"Encountered {len(other_errors)} non-duplicate errors during bulk index: {other_errors}")

        except exceptions.ConnectionError as e:
            print(f"ES connection error during bulk index: {e}")
        except exceptions.TransportError as e:
            print(f"ES transport error during bulk index: status={e.status_code}, error={e.error}")
        except Exception as e:
            print(f"Unexpected indexing error: {e}")

def process_transcript(path: str, size: int, step: int, index_name: str, meta_rss_csv: str) -> int:
    """
    Worker to chunk, embed, and index a single transcript file.
    """
    complete_chunks = create_complete_chunks(path, size, step, meta_rss_csv)
    index(index_name, complete_chunks)
    return len(complete_chunks)


def prepare_path_lists(index_name:str, category:str, media_directory:str, limit: None):
    if es.indices.exists(index=index_name):
        res = es.search(index=index_name, query={"match_all": {}}, size=9999, _source=[category])
        episode_names = [hit["_source"][category] for hit in res["hits"]["hits"]]
        episode_names_es = set(episode_names)
    else:
        episode_names_es = set()

    media_directory_list = glob.glob(media_directory)
    episode_names_in_directory = set([Path(i).stem for i in media_directory_list])

    transcripts_to_process = list(episode_names_in_directory - episode_names_es)

    transcript_to_process_list = []
    print(f"Total number of transcripts left to process: {len(transcripts_to_process)}")
    for i in transcripts_to_process:
        f = f"ingest-data/media-files/huberman/{i}.txt"
        transcript_to_process_list.append(f)

    if limit is not None:
        transcript_to_process_list = transcript_to_process_list[:limit]

    return transcript_to_process_list

def index_multiple_transcripts(size:int, step:int, index_name:str, meta_rss_csv:str, 
                               category:str, media_directory:str, limit:int=None):
    """
    Chunk, embed, and index multiple transcripts sequentially.
    """
    total_indexed = 0
    transcript_indexed = 0

    path_list = prepare_path_lists(index_name, category, media_directory, limit=limit)
    for transcript_path in path_list:
        try:
            indexed_count = process_transcript(transcript_path, size, step, index_name, meta_rss_csv)
            total_indexed += indexed_count
            transcript_indexed += 1
            print(f"Indexed {indexed_count} chunks from {transcript_path}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Failed to index {transcript_path}: {exc}")
        print(f"Number of transcripts processed in current run: {transcript_indexed} / {len(path_list)}")

    print(f"Total chunks indexed: {total_indexed}")
    print(f"Total number of transcripts indexed: {transcript_indexed}")
    return "SUCCESS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk, embed, and index transcripts into Elasticsearch.")
    parser.add_argument(
        "--size",
        type=int,
        default=200,
        help="Number of words per chunk."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=150,
        help="Sliding window step size."
    )
   
    parser.add_argument(
        "--meta-rss-csv",
        required=True,
        help="Path to RSS metadata CSV with ep_name, name_of_podcast, id columns."
    )
    parser.add_argument(
        "--index-name",
        required=True,
        help="Elasticsearch index name to write chunks into."
    )
    parser.add_argument(
        "--category",
        default="episode_name",
        help="Document field to compare against transcript filenames for filtering (e.g., 'episode_name')."
    )
    parser.add_argument(
        "--media-directory",
        required=True,
        help="Glob pattern for transcript files (e.g., 'ingest-data/media-files/*.txt')."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of transcripts to process."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    index_multiple_transcripts(
        size=args.size,
        step=args.step,
        index_name=args.index_name,
        meta_rss_csv=args.meta_rss_csv,
        category=args.category,
        media_directory=args.media_directory,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
