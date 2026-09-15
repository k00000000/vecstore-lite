"""vecstore-lite：嵌入式 SQLite + NumPy 向量库。"""

from .embedding import hash_embed, random_projection
from .store import SearchHit, VecStore

__all__ = [
    "VecStore",
    "SearchHit",
    "hash_embed",
    "random_projection",
    "__version__",
]
__version__ = "1.1.0"
