"""
GAMESA/KrystalStack - Cognitive Stream (Python)

This module implements the Cognitive Stream components:
- MetacognitiveInterface: Self-reflecting analysis of system performance
- PolicyProposalGenerator: LLM-driven policy proposals
- ExperienceStore: State-Action-Reward storage and retrieval
"""

from .metacognitive import MetacognitiveInterface
from .experience_store import ExperienceStore
from .policy_generator import PolicyProposalGenerator

__all__ = [
    "MetacognitiveInterface",
    "ExperienceStore",
    "PolicyProposalGenerator",
]
