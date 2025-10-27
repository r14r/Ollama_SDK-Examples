import streamlit as st

from lib.helper_ollama import helper


def add_select_temperature(label: str = "Temperature", default: float = 0.7):
    """Add a temperature slider to the Streamlit sidebar."""

    return st.slider(
        label,
        min_value=0.0,
        max_value=2.0,
        value=default,
        step=0.1,
        help="""Controls randomness in generation.
         Higher values make output more random.
         Lower temperature for more consistent structured outputs.
         """,
    )


def add_select_max_tokens(label: str = "Max Tokens", default: int = 1024):
    """Add a max tokens number input to the Streamlit sidebar."""

    return st.number_input(
        label,
        min_value=1,
        max_value=4096,
        value=default,
        help="Maximum number of tokens in response",
    )


def add_select_top_p(label: str = "Top P", default: float = 0.9):
    return st.slider(
        label,
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.1,
        help="Nucleus sampling parameter",
    )


def add_select_models():
    return st.selectbox(
        "Select Model",
        options=helper.models_list(with_details=False),
        help="Choose a model that supports tool calling",
    )

def add_select_models_tooling():
    return st.selectbox(
        "Select Model",
        options=helper.models_list(with_details=False),
        help="Choose a model that supports tool calling",
    )

def add_select_models_vision():
    vision_models = ["gemma3", "llama3.2-vision", "llava", "bakllava"]

    return st.selectbox(
            "Select Vision Model",
            options=vision_models,
            help="Choose a model that supports vision/multimodal input"
        )

def add_select_models_embedding():
    """Add a select box for choosing an embedding model."""
    embedding_models = [
        "text-embedding-ada-002",
        "text-embedding-babbage-001",
        "text-embedding-curie-001",
        "text-embedding-davinci-001",
    ]
    embedding_models = [
        "llama3.2",
        "nomic-embed-text",
        "all-minilm",
        "sentence-transformers",
    ]

    return st.selectbox(
        "Select Embedding Model",
        options=embedding_models,
        help="Choose a model that supports embeddings",
    )
