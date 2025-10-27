"""
Pages module for Ollama Streamlit Demo

This module contains all the individual page implementations for the demo app.
"""

# Import all page modules
from . import (
    Chat,
    Embeddings,
    Generation,
    Model_Management,
    Multimodal,
    Streaming,
    Structured_Outputs,
    Tools
)

__all__ = [
    'Chat',
    'Generation', 
    'Multimodal',
    'Embeddings',
    'Model_Management',
    'Tools',
    'Structured_Outputs',
    'Streaming'
]