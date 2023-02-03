from sentence_transformers import SentenceTransformer
from fastapi import APIRouter, Body

from schemas.emebeddings import EmbeddingsRequest, EmbeddingsResponse

router = APIRouter(prefix="/embeddings")

# choose here various pretrained english or multi-lingual models from
# https://www.sbert.net/docs/pretrained_models.html
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


@router.post("/", response_model=EmbeddingsResponse)
async def embed(
    payload: EmbeddingsRequest = Body(
        ..., example={"strings": ["Hello", "你好", "Bonjour", "今日は", "नमस्ते", "Hallo", "Hola"]}
    )
) -> EmbeddingsResponse:
    """Embeds a list of strings, returning a list of floats for every string input"""
    embeddings = [i.tolist() for i in model.encode(payload.strings)]
    return EmbeddingsResponse(embeddings=embeddings, **payload.dict())
