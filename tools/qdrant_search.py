import os
import re
from threading import Lock
from typing import Callable, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


IntLike = Union[str, int, float]


class QdrantSearchClient:

    DEFAULT_RESULTS = 15
    DEFAULT_CANDIDATES = 100

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_cloud: Optional[bool] = None,
        cloud_api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        self.payload_fields = ["chunk", "episode_name", "start", "end"]
        self.default_num_results = self.DEFAULT_RESULTS
        self.default_num_candidates = self.DEFAULT_CANDIDATES

        model_name = "all-MPNet-base-v2"
        self.embedding_model = SentenceTransformer(model_name)
        self._embed_lock = Lock()

        # Decide whether to use a cloud deployment vs the local docker container.

        provided_api_key = os.getenv("QDRANT_API_KEY")
        resolved_cloud_url = os.getenv("QDRANT_ENDPOINT")
        resolved_cloud_key = cloud_api_key or provided_api_key

        cloud_requested = bool(use_cloud)
        cloud_available = bool(resolved_cloud_url and resolved_cloud_key)

        if cloud_requested and not cloud_available:
            raise ValueError(
                "Cloud Qdrant requested but QDRANT_ENDPOINT and QDRANT_API_KEY are missing."
            )

        if cloud_requested or (not cloud_requested and cloud_available):
            self.mode = "cloud"
            self.client = QdrantClient(
                url=resolved_cloud_url,
                api_key=resolved_cloud_key,
                prefer_grpc=prefer_grpc,
            )
        else:
            self.mode = "local"
            host = host or "127.0.0.1"
            port = port or 6333
            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=provided_api_key,
                prefer_grpc=prefer_grpc,
            )

    def embed_query(self, query: str) -> List[float]:
        if not query:
            raise ValueError("Query text cannot be empty.")
        with self._embed_lock:
            vector = self.embedding_model.encode(query, convert_to_numpy=True)
        if hasattr(vector, "tolist"):
            return vector.tolist()
        if isinstance(vector, list):
            return vector
        return list(vector)

    @staticmethod
    def _coerce_positive_int(value: Optional[IntLike], default: int) -> int:
        """
        Convert user/tool supplied values into positive integers.
        Falls back to default when conversion fails.
        """
        if value is None:
            return default
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            try:
                coerced = int(value)
            except (TypeError, ValueError):
                return default
            return max(coerced, 1)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            # Extract the first integer found in the string if possible.
            match = re.search(r"-?\d+", stripped)
            if match:
                try:
                    coerced = int(match.group())
                    return max(coerced, 1)
                except (TypeError, ValueError):
                    return default
            return default
        return default

    def search_embeddings(
        self,
        query: str,
        collection_name="transcripts",
        num_results: Optional[IntLike] = 15,
        num_candidates: Optional[IntLike] = 100,
    ) -> List[dict[str, str]]:
        
        if not query:
            raise ValueError("query is required.")

        limit = self._coerce_positive_int(num_results, self.default_num_results)
        candidates = self._coerce_positive_int(
            num_candidates, max(limit * 5, self.default_num_candidates)
        )
        vector = self.embed_query(query)

        params = qmodels.SearchParams(hnsw_ef=candidates, exact=False)
        response = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=self.payload_fields,
            search_params=params,
        )

        points = getattr(response, "result", response)

        results: List[dict[str, str]] = []
        for point in points:
            payload: dict[str, str] = {}
            score = None

            if isinstance(point, tuple):
                raw_point, score = point[0], point[1]
            else:
                raw_point = point
                score = getattr(raw_point, "score", None)

            if hasattr(raw_point, "payload"):
                payload = raw_point.payload or {}
            elif isinstance(raw_point, dict):
                payload = raw_point
            else:
                # Fallback when only raw text/strings are returned; treat as chunk.
                payload = {"chunk": str(raw_point)}
            results.append(
                {
                    "episode_name": payload.get("episode_name", ""),
                    "start": payload.get("start", ""),
                    "end": payload.get("end", ""),
                    "chunk": payload.get("chunk", ""),
                    "score": score,
                }
            )
        return results


def create_qdrant_search_tools(
    **client_kwargs,
) -> tuple[
    Callable[[str], List[float]],
    Callable[[str, Optional[str], Optional[IntLike], Optional[IntLike]], List[dict[str, str]]],
]:
    """
    Factory that returns embedding and search callables backed by Qdrant.
    """
    client = QdrantSearchClient(**client_kwargs)
    def embedding(query: str) -> List[float]:
        vector = client.embed_query(query)
        if hasattr(vector, "tolist"):
            return vector.tolist()
        if isinstance(vector, list):
            return vector
        return list(vector)
    embedding.__name__ = "embed_query"

    def search(
        query: str,
        collection_name: Optional[str] = None,
        num_results: Optional[IntLike] = None,
        num_candidates: Optional[IntLike] = None,
    ) -> List[dict[str, str]]:
        target_collection = (
            collection_name
            or client.collection_name
            if hasattr(client, "collection_name")
            else "transcripts"
        )
        return client.search_embeddings(
            query=query,
            collection_name=target_collection,
            num_results=client._coerce_positive_int(
                num_results, client.default_num_results
            ),
            num_candidates=client._coerce_positive_int(
                num_candidates, client.default_num_candidates
            ),
        )
    search.__name__ = "vector_search"

    return embedding, search
