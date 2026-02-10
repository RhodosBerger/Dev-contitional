"""
Log Ingestion Service (Producer) 📤
Reads logs and documents, pushes them to Kafka.
"""
import logging
import json
import time
from datetime import datetime, timezone
from .kafka_client import KafkaClient

logger = logging.getLogger("RAG.Producer")

class LogIngestionService:
    def __init__(self, kafka_client: KafkaClient):
        self.kafka = kafka_client
        self.topic = "rag-ingestion"
        self.kafka.connect_producer()

    def ingest_log(self, log_entry: dict):
        """Pushes a log entry to Kafka."""
        if not log_entry.get('timestamp'):
            log_entry['timestamp'] = datetime.now(timezone.utc).isoformat()
            
        logger.info(f"Ingesting log: {log_entry.get('timestamp')} - {log_entry.get('message')}")
        payload = json.dumps({"type": "LOG", "data": log_entry})
        self.kafka.producer.produce(self.topic, payload)

    def ingest_document(self, doc_id: str, content: str, metadata: dict = None):
        """Pushes a document to Kafka."""
        logger.info(f"Ingesting document: {doc_id}")
        payload = json.dumps({
            "type": "DOCUMENT", 
            "id": doc_id, 
            "content": content,
            "metadata": metadata or {}
        })
        self.kafka.producer.produce(self.topic, payload)
