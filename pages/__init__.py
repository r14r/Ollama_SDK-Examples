"""
Pages module for Ollama Streamlit Demo

This module contains all the individual page implementations for the demo app.
"""

# Import all page modules
from . import Chat
from . import Embeddings
from . import Generation
from . import Model_Management
from . import Multimodal
from . import Streaming
from . import Structured_Outputs
from . import Tools

__all__ = [
    "Chat",
    "Generation",
    "Multimodal",
    "Embeddings",
    "Model_Management",
    "Tools",
    "Structured_Outputs",
    "Streaming",
]
