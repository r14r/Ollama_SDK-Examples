import sys
from pathlib import Path

import streamlit as st
from lib.helper_ollama import helper

lib_path = Path(__file__).parent / "lib"
if str(lib_path) not in sys.path:
    sys.path.insert(0, str(lib_path))

from pages import (  # noqa: E402
    Chat,
    Embeddings,
    Generation,
    Model_Management,
    Multimodal,
    Streaming,
    Structured_Outputs,
    Tools,
)


def main():
    """Main Streamlit app function."""

    st.set_page_config(
        page_title="Ollama SDK Demo",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar for navigation
    st.sidebar.title("🤖 Ollama SDK Demo")
    st.sidebar.markdown("---")

    # Page selection
    pages = {
        "🏠 Home": "home",
        "💬 Chat Demo": "chat",
        "📝 Text Generation": "generation",
        "🖼️ Multimodal Chat": "multimodal",
        "🔗 Embeddings": "embeddings",
        "⚙️ Model Management": "models",
        "🛠️ Tool Calling": "tools",
        "📊 Structured Outputs": "structured",
        "⚡ Streaming Demo": "streaming",
    }

    selected_page = st.sidebar.selectbox("Select a page:", options=list(pages.keys()), index=0)

    page_key = pages[selected_page]

    # Display the selected page
    if page_key == "home":
        show_home_page()
    elif page_key == "chat":
        Chat.show()
    elif page_key == "generation":
        Generation.show()
    elif page_key == "multimodal":
        Multimodal.show()
    elif page_key == "embeddings":
        Embeddings.show()
    elif page_key == "models":
        Model_Management.show()
    elif page_key == "tools":
        Tools.show()
    elif page_key == "structured":
        Structured_Outputs.show()
    elif page_key == "streaming":
        Streaming.show()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.markdown("This app demonstrates all features of the Ollama Python SDK " "using the OllamaHelper class extracted from the examples.")

    # Quick model status
    with st.sidebar.expander("🔍 Quick Model Check"):
        if st.button("List Available Models"):
            with st.spinner("Fetching models..."):
                try:
                    models = helper.models_list(with_details=True)
                    st.write(f"**{len(models)} models available:**")
                    for model in models[:5]:  # Show first 5
                        st.write(f"• {model.model}")
                    if len(models) > 5:
                        st.write(f"... and {len(models) - 5} more")
                except Exception as e:
                    st.error(f"Error: {e}")


def show_home_page():
    """Display the home page."""

    st.title("🤖 Ollama Python SDK Demo")
    st.markdown("### Welcome to the comprehensive Ollama SDK demonstration!")

    st.markdown(
        """
    This Streamlit application showcases all the functionality available in the Ollama Python SDK
    through our custom `OllamaHelper` class. Each page demonstrates different capabilities:
    """
    )

    # Feature overview
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        #### 🔥 Core Features
        - **💬 Chat Operations** - Basic and advanced chat functionality
        - **📝 Text Generation** - Simple and streaming text generation
        - **🖼️ Multimodal Chat** - Chat with images and visual content
        - **🔗 Embeddings** - Text embeddings for similarity and search
        """
        )

    with col2:
        st.markdown(
            """
        #### ⚙️ Advanced Features
        - **⚙️ Model Management** - List, pull, and manage models
        - **🛠️ Tool Calling** - Function calling with AI models
        - **📊 Structured Outputs** - JSON schema and Pydantic models
        - **⚡ Streaming** - Real-time streaming responses
        """
        )

    st.markdown("---")

    # Quick start section
    st.markdown("### 🚀 Quick Start")

    # Model selection for quick test
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        model = st.selectbox(
            "Select a model for quick test:",
            options=["gemma3", "llama3.1", "llama3.2", "codellama"],
            help="Choose a model that you have installed",
        )

    with col2:
        if st.button("🧪 Test Model", type="primary"):
            with st.spinner(f"Testing {model}..."):
                try:
                    response = st.session_state.helper.simple_chat(model, "Say hello and tell me you're working!")
                    st.success("✅ Model is working!")
                    st.info(f"**Response:** {response}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    with col3:
        if st.button("📋 List Models"):
            with st.spinner("Fetching models..."):
                try:
                    models = st.session_state.helper.models_list(with_details=True)
                    st.write(f"Found {len(models.models)} models:")
                    for model in models.models:
                        size_mb = model.size.real / 1024 / 1024
                        st.write(f"• **{model.model}** ({size_mb:.1f} MB)")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")

    # Statistics and info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Demo Pages", "8", help="Total number of feature demo pages")

    with col2:
        st.metric("🔧 Helper Methods", "30+", help="Methods available in OllamaHelper class")

    with col3:
        st.metric("📚 Examples", "25+", help="Based on official Ollama examples")

    st.markdown("---")

    # Tips section
    st.markdown("### 💡 Tips for Using This Demo")

    tips = [
        "**Start with Chat Demo** - Get familiar with basic functionality",
        "**Check Model Management** - Ensure you have models installed",
        "**Try Streaming Demo** - See real-time responses",
        "**Explore Tool Calling** - Advanced AI function calling",
        "**Test Multimodal** - Upload images for visual AI",
        "**Use Structured Outputs** - Get JSON responses",
    ]

    for tip in tips:
        st.markdown(f"• {tip}")

    st.markdown("---")

    # Code example
    with st.expander("📋 Example Code Usage"):
        st.code(
            """
# Initialize the helper
from lib.helper_ollama import helper

# Simple chat
response = helper.simple_chat('gemma3', 'Hello, how are you?')
print(response)

# Generate text
text = helper.simple_generate('gemma3', 'Write a haiku about coding')
print(text)

# Get embeddings
embeddings = helper.get_embeddings('llama3.2', 'Hello world')
print(f"Embedding dimensions: {len(embeddings)}")

# List available models
models = helper.models_list()
for model in models.models:
    print(f"Model: {model.model}")
        """,
            language="python",
        )


if __name__ == "__main__":
    main()
