from typing import Tuple
from pathlib import Path

import requests
import numpy as np
from joblib import Memory
from scipy.spatial.distance import cdist
from wordfreq import top_n_list
from gensim.models import KeyedVectors


memory = Memory(".cache", verbose=0)


@memory.cache
def load_words() -> Tuple[np.ndarray, np.ndarray]:
    """Loads the top 100,000 English words and their embeddings."""
    path = Path("data/embeddings/GoogleNews-vectors-negative300.bin")
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    words = top_n_list("en", 100_000)
    words = np.array([word for word in words if word in model])
    embeddings = np.array([model[word] for word in words])
    return words, embeddings


@memory.cache
def get_rank(word: str, game: int) -> (int | None):
    """Returns the rank of a word in a Contexto game."""
    response = requests.get(f"https://api.contexto.me/machado/en/game/{game}/{word}")
    response = response.json()
    rank = response.get("distance", None)
    return rank


def get_distances(embeddings: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Returns the cosine distances between the `embeddings` and a `e`."""
    return cdist(embeddings, e.reshape(1, -1), metric='cosine').flatten()