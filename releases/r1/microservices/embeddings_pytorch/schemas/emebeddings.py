from typing import List

from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    strings: List[str]


class EmbeddingsResponse(EmbeddingsRequest):
    embeddings: List[List[float]]
