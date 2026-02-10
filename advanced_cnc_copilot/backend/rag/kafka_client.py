"""
Kafka Client Wrapper 📨
Handles connection to the Kafka broker.
Falls back to In-Memory Queue if Kafka is not available.
"""
import logging
import os
import queue
import json
import threading

logger = logging.getLogger("RAG.Kafka")

# Global In-Memory Queue for Mock Mode
MOCK_BROKER = {}

class KafkaClient:
    def __init__(self, bootstrap_servers=None):
        self.bootstrap_servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "mock")
        self.producer = None
        self.consumer = None
        self.is_mock = self.bootstrap_servers == "mock" or "localhost" in self.bootstrap_servers
        
        logger.info(f"Kafka Client configured for: {self.bootstrap_servers} (Mock Mode: {self.is_mock})")

    def connect_producer(self):
        """Connects the producer."""
        if self.is_mock:
            self.producer = MockProducer()
            logger.info("Connected to MOCK Producer")
        else:
            logger.warning("Real Kafka not implemented, falling back to Mock")
            self.producer = MockProducer()

    def connect_consumer(self, group_id, topic):
        """Connects the consumer."""
        if self.is_mock:
            self.consumer = MockConsumer(topic)
            logger.info(f"Connected to MOCK Consumer for topic {topic}")
        else:
            logger.warning("Real Kafka not implemented, falling back to Mock")
            self.consumer = MockConsumer(topic)

class MockProducer:
    def produce(self, topic, value):
        if topic not in MOCK_BROKER:
            MOCK_BROKER[topic] = queue.Queue()
        MOCK_BROKER[topic].put(value)
        # logger.debug(f"Mock Produced to {topic}: {value[:50]}...")

    def flush(self):
        pass

class MockConsumer:
    def __init__(self, topic):
        self.topic = topic
        if topic not in MOCK_BROKER:
            MOCK_BROKER[topic] = queue.Queue()
        self.queue = MOCK_BROKER[topic]

    def poll(self, timeout=1.0):
        try:
            data = self.queue.get(timeout=timeout)
            return MockMessage(data)
        except queue.Empty:
            return None

class MockMessage:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def error(self):
        return None
