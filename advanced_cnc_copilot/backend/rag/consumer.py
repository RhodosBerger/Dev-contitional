"""
Embedding Worker (Consumer) 📥
Reads from Kafka, generates embeddings, stores in VectorDB.
"""
import logging
import json
import time
import threading
import random
from .kafka_client import KafkaClient
from .vector_store import VectorStore

logger = logging.getLogger("RAG.Worker")

class EmbeddingWorker:
    def __init__(self, kafka_client: KafkaClient, vector_store: VectorStore):
        self.kafka = kafka_client
        self.store = vector_store
        self.running = False
        self.topic = "rag-ingestion"
        self.kafka.connect_consumer("rag-group", self.topic)

    def _generate_mock_embedding(self, text: str):
        """
        Generates a deterministic mock embedding based on text hash.
        Pure Python implementation.
        """
        # Create a seed from string hash
        seed = abs(hash(text))
        random.seed(seed)
        
        # Generate 64-dim vector (small for mock)
        return [random.uniform(-1.0, 1.0) for _ in range(64)]

    def _process_message(self):
        while self.running:
            msg = self.kafka.consumer.poll(timeout=1.0)
            if msg is None:
                continue
                
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue
                
            try:
                payload = json.loads(msg.value())
                doc_type = payload.get("type")
                
                if doc_type == "DOCUMENT":
                    doc_id = payload.get("id")
                    content = payload.get("content")
                    metadata = payload.get("metadata", {})
                    
                    logger.info(f"Processing document {doc_id}...")
                    embedding = self._generate_mock_embedding(content)
                    
                    self.store.add(doc_id, embedding, {**metadata, "content_snippet": content[:50]})
                    logger.info(f"Document {doc_id} indexed.")
                    
                elif doc_type == "LOG":
                    data = payload.get("data", {})
                    log_id = f"LOG-{int(time.time()*1000)}"
                    content = data.get("message", "")
                    
                    embedding = self._generate_mock_embedding(content)
                    self.store.add(log_id, embedding, {"source": "log", **data})
                    
            except Exception as e:
                logger.error(f"Failed to process message: {e}")

    def start(self):
        """Starts the consumption loop in a background thread."""
        if self.running:
            return
            
        self.running = True
        logger.info("Starting Embedding Worker...")
        self.thread = threading.Thread(target=self._process_message, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        logger.info("Embedding Worker stopped.")
