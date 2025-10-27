"""
Streaming Demo Page

Demonstrates real-time streaming functionality using OllamaHelper.
"""

import streamlit as st
import time

from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_models_tooling, add_select_temperature


def show():
    """Display the streaming demo page."""
    
    st.title("⚡ Streaming Demo")
    st.markdown("Experience real-time streaming responses from Ollama models.")
    
    # Get helper from session state
    helper = OllamaHelper()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Streaming Settings")

        model = add_select_models_tooling()
        temperature = add_select_temperature(default=0.7)
        
        stream_mode = st.radio(
            "Stream Mode",
            options=["Chat Streaming", "Generation Streaming", "Story Writing"],
            help="Choose streaming demonstration type"
        )
        
        show_tokens = st.checkbox("Show individual tokens", value=False)
        delay_simulation = st.checkbox("Simulate typing delay", value=True)
    
    # Main content area
    if stream_mode == "Chat Streaming":
        show_chat_streaming(helper, model, temperature, show_tokens, delay_simulation)
    elif stream_mode == "Generation Streaming":
        show_generation_streaming(helper, model, temperature, show_tokens, delay_simulation)
    else:
        show_story_streaming(helper, model, temperature, show_tokens, delay_simulation)


def show_chat_streaming(helper, model: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Show streaming chat interface."""
    
    st.subheader("💬 Streaming Chat")
    st.markdown("Watch responses appear in real-time as they're generated.")
    
    # Input
    user_message = st.text_area(
        "Your message:",
        placeholder="Ask me anything...",
        height=100
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🚀 Stream Response", type="primary"):
            if user_message.strip():
                stream_chat_response(helper, model, user_message, temperature, show_tokens, delay_simulation)
            else:
                st.warning("Please enter a message.")
    
    with col2:
        if st.button("🎲 Try Example Question"):
            examples = [
                "Explain how machine learning works",
                "Write a Python function to reverse a string",
                "What are the benefits of renewable energy?",
                "Describe the process of photosynthesis",
                "How do quantum computers work?"
            ]
            import random
            user_message = random.choice(examples)
            st.rerun()


def show_generation_streaming(helper, model: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Show streaming text generation interface."""
    
    st.subheader("📝 Streaming Text Generation")
    st.markdown("Generate long-form text with real-time streaming.")
    
    # Prompt templates
    prompt_templates = {
        "Essay": "Write a detailed essay about",
        "Article": "Write an informative article on",
        "Tutorial": "Create a step-by-step tutorial for",
        "Analysis": "Provide an in-depth analysis of",
        "Review": "Write a comprehensive review of",
        "Custom": ""
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        template_type = st.selectbox("Choose prompt template:", list(prompt_templates.keys()))
        
        if template_type == "Custom":
            prompt = st.text_area("Custom prompt:", height=100)
        else:
            topic = st.text_input("Topic:", placeholder="artificial intelligence, climate change, etc.")
            prompt = f"{prompt_templates[template_type]} {topic}" if topic else ""
    
    with col2:
        st.markdown("**Template Examples:**")
        st.write("📄 Essay - Academic writing")
        st.write("📰 Article - News/blog style")
        st.write("📚 Tutorial - How-to guide")
        st.write("🔍 Analysis - Deep dive")
        st.write("⭐ Review - Evaluation")
    
    if st.button("✨ Generate Streaming Text", type="primary"):
        if prompt.strip():
            stream_generation_response(helper, model, prompt, temperature, show_tokens, delay_simulation)
        else:
            st.warning("Please enter a prompt.")


def show_story_streaming(helper, model: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Show interactive story streaming."""
    
    st.subheader("📖 Interactive Story Streaming")
    st.markdown("Watch a story unfold in real-time with interactive elements.")
    
    # Story setup
    story_elements = {
        "Genre": ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Adventure", "Horror"],
        "Setting": ["Medieval castle", "Space station", "Modern city", "Enchanted forest", "Desert island", "Underground lab"],
        "Character": ["Brave knight", "Space explorer", "Detective", "Wizard", "Pirate", "Scientist"],
        "Conflict": ["Ancient curse", "Alien invasion", "Missing treasure", "Time paradox", "Natural disaster", "Secret conspiracy"]
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        genre = st.selectbox("Genre:", story_elements["Genre"])
        setting = st.selectbox("Setting:", story_elements["Setting"])
    
    with col2:
        character = st.selectbox("Main Character:", story_elements["Character"])
        conflict = st.selectbox("Central Conflict:", story_elements["Conflict"])
    
    # Custom elements
    custom_element = st.text_input("Add custom element (optional):", placeholder="magical sword, talking animal, etc.")
    
    story_prompt = f"Write an engaging {genre.lower()} story set in a {setting.lower()}. The main character is a {character.lower()} who faces {conflict.lower()}."
    if custom_element:
        story_prompt += f" Include a {custom_element} in the story."
    story_prompt += " Make it captivating with vivid descriptions and dialogue."
    
    st.markdown("**Story Prompt:**")
    st.info(story_prompt)
    
    if st.button("📚 Begin Story Streaming", type="primary"):
        stream_story_response(helper, model, story_prompt, temperature, show_tokens, delay_simulation)


def stream_chat_response(helper, model: str, message: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Stream a chat response with real-time display."""
    
    st.subheader("🤖 Streaming Response:")
    
    # Create placeholder for streaming content
    response_placeholder = st.empty()
    accumulated_response = ""
    
    try:
        messages = [{'role': 'user', 'content': message}]
        
        # Use the helper's streaming method
        for chunk in helper.chat_stream(model, messages, options={'temperature': temperature}):
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                accumulated_response += content
                
                # Display with formatting
                if show_tokens:
                    response_placeholder.markdown(f"**Tokens:** `{repr(content)}`\n\n**Response:** {accumulated_response}")
                else:
                    response_placeholder.markdown(accumulated_response)
                
                # Simulate typing delay
                if delay_simulation and content:
                    time.sleep(0.02)
        
        # Final response with metadata
        st.markdown("---")
        with st.expander("📊 Stream Statistics"):
            st.write(f"**Model:** {model}")
            st.write(f"**Temperature:** {temperature}")
            st.write(f"**Total Characters:** {len(accumulated_response)}")
            st.write(f"**Word Count:** {len(accumulated_response.split())}")
    
    except Exception as e:
        st.error(f"Streaming error: {e}")


def stream_generation_response(helper, model: str, prompt: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Stream a generation response with real-time display."""
    
    st.markdown("### 📄 Streaming Generation:")
    
    # Create placeholder for streaming content
    response_placeholder = st.empty()
    accumulated_response = ""
    word_count = 0
    
    try:
        # Use the helper's streaming method
        for chunk in helper.generate_stream(model, prompt, options={'temperature': temperature}):
            if 'response' in chunk:
                content = chunk['response']
                accumulated_response += content
                word_count = len(accumulated_response.split())
                
                # Update display
                if show_tokens:
                    response_placeholder.markdown(f"**Tokens:** `{repr(content)}`\n\n**Generated Text:** {accumulated_response}")
                else:
                    response_placeholder.markdown(accumulated_response)
                
                # Show progress
                if word_count > 0 and word_count % 50 == 0:
                    st.sidebar.success(f"Words generated: {word_count}")
                
                # Simulate typing delay
                if delay_simulation and content:
                    time.sleep(0.03)
        
        # Final statistics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", len(accumulated_response))
        with col2:
            st.metric("Words", len(accumulated_response.split()))
        with col3:
            st.metric("Sentences", accumulated_response.count('.') + accumulated_response.count('!') + accumulated_response.count('?'))
    
    except Exception as e:
        st.error(f"Streaming error: {e}")


def stream_story_response(helper, model: str, prompt: str, temperature: float, show_tokens: bool, delay_simulation: bool):
    """Stream a story with enhanced presentation."""
    
    st.markdown("### 📖 Your Story Unfolds:")
    
    # Create placeholder for streaming content
    response_placeholder = st.empty()
    accumulated_response = ""
    sentence_count = 0
    
    try:
        # Use higher temperature for creative writing
        creative_temp = min(temperature + 0.3, 2.0)
        
        for chunk in helper.generate_stream(model, prompt, options={'temperature': creative_temp}):
            if 'response' in chunk:
                content = chunk['response']
                accumulated_response += content
                
                # Count sentences for dramatic pauses
                sentence_count = accumulated_response.count('.') + accumulated_response.count('!') + accumulated_response.count('?')
                
                # Display with story formatting
                formatted_story = accumulated_response.replace('\n\n', '\n\n---\n\n')
                
                if show_tokens:
                    response_placeholder.markdown(f"**Tokens:** `{repr(content)}`\n\n{formatted_story}")
                else:
                    response_placeholder.markdown(formatted_story)
                
                # Dramatic pauses at sentence endings
                if delay_simulation:
                    if content in '.!?':
                        time.sleep(0.5)  # Longer pause at sentence end
                    elif content == ',':
                        time.sleep(0.2)  # Short pause at comma
                    else:
                        time.sleep(0.04)  # Regular character delay
        
        # Story completion
        st.markdown("---")
        st.success("📚 Story Complete!")
        
        # Story analysis
        with st.expander("📊 Story Analysis"):
            words = accumulated_response.split()
            sentences = sentence_count
            paragraphs = accumulated_response.count('\n\n') + 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", len(words))
            with col2:
                st.metric("Sentences", sentences)
            with col3:
                st.metric("Paragraphs", paragraphs)
            with col4:
                reading_time = len(words) / 200  # Assume 200 WPM
                st.metric("Reading Time", f"{reading_time:.1f} min")
    
    except Exception as e:
        st.error(f"Streaming error: {e}")


# Tips section
def show_streaming_tips():
    """Show tips for streaming."""
    with st.expander("💡 Streaming Tips"):
        st.markdown("""
        **Streaming Benefits:**
        - Real-time feedback and engagement
        - Better user experience for long responses
        - Ability to stop generation early
        - Immediate indication that the model is working
        
        **Best Practices:**
        - Use appropriate delays for readability
        - Show progress indicators for long generations
        - Allow users to interrupt streaming
        - Handle connection errors gracefully
        
        **Model Performance:**
        - Smaller models stream faster
        - Network latency affects streaming speed
        - Complex prompts may have irregular streaming
        """)


if __name__ == "__main__":
    show()