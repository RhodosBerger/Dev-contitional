"""
Vector Store Interface 🗄️
Implements a lightweight In-Memory Vector Store.
Supports NumPy but falls back to pure Python if unavailable.
"""
import logging
import json
import os
import math
from typing import List, Dict, Any

logger = logging.getLogger("RAG.VectorStore")

try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False
    logger.warning("NumPy not found. Using pure Python fallback for Vector Store.")

class VectorStore:
    def __init__(self, path="./vector_db.json"):
        self.path = path
        self.documents = []
        self.vectors = None
        self._load()
        logger.info(f"Vector Store initialized. Loaded {len(self.documents)} documents.")

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    # If using numpy, reconstruct array
                    if USE_NUMPY and self.documents:
                        self.vectors = np.array([doc['embedding'] for doc in self.documents], dtype=np.float32)
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")

    def _save(self):
        try:
            # Ensure embeddings are lists for JSON
            serializable_docs = []
            for doc in self.documents:
                doc_copy = doc.copy()
                if USE_NUMPY and isinstance(doc_copy.get('embedding'), np.ndarray):
                    doc_copy['embedding'] = doc_copy['embedding'].tolist()
                serializable_docs.append(doc_copy)
            
            with open(self.path, 'w') as f:
                json.dump({"documents": serializable_docs}, f)
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def add(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]):
        """Adds a vector to the store."""
        if USE_NUMPY:
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            # Store mock vector
            self.documents.append({"id": doc_id, "embedding": vec, "metadata": metadata})
            
            if self.vectors is None:
                self.vectors = vec.reshape(1, -1)
            else:
                self.vectors = np.vstack([self.vectors, vec])
        else:
            # Pure Python Sync
            # Normalize
            msg_mag = math.sqrt(sum(x*x for x in embedding))
            if msg_mag > 0:
                vec = [x/msg_mag for x in embedding]
            else:
                vec = embedding
            self.documents.append({"id": doc_id, "embedding": vec, "metadata": metadata})
            
        self._save()
        logger.info(f"Stored vector for {doc_id}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches for similar vectors."""
        if not self.documents:
            return []

        if USE_NUMPY:
            if self.vectors is None: return []
            query_vec = np.array(query_embedding, dtype=np.float32)
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            scores = np.dot(self.vectors, query_vec)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                results.append({
                    "id": doc["id"],
                    "score": float(scores[idx]),
                    "metadata": doc["metadata"]
                })
            return results
        else:
            # Pure Python Cosine Similarity
            query_mag = math.sqrt(sum(x*x for x in query_embedding))
            if query_mag == 0: return []
            
            results = []
            for doc in self.documents:
                vec = doc["embedding"]
                dot = sum(a*b for a,b in zip(vec, query_embedding))
                # vec is already normalized in add()
                score = dot / query_mag
                results.append({
                    "id": doc["id"],
                    "score": score,
                    "metadata": doc["metadata"]
                })
            
            # Sort desc
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
