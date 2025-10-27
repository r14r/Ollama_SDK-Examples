"""
Chat Demo Page

Demonstrates basic and advanced chat functionality using OllamaHelper.
"""

import streamlit as st
from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_max_tokens
from lib.helper_ui import add_select_models_tooling
from lib.helper_ui import add_select_temperature


def show():
    """Display the chat demo page."""

    st.title("💬 Chat Demo")
    st.markdown("Explore different chat modes with Ollama models.")

    # Get helper from session state
    helper = OllamaHelper()

    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Chat Settings")

        model = add_select_models_tooling()
        temperature = add_select_temperature(default=0.7)
        max_tokens = add_select_max_tokens(label="Max Tokens", default=1024)

        chat_mode = st.radio(
            "Chat Mode",
            options=["Simple Chat", "Conversation", "System Prompt"],
            help="Choose chat interaction mode",
        )

    # Main content area
    if chat_mode == "Simple Chat":
        show_simple_chat(helper, model, temperature, max_tokens)
    elif chat_mode == "Conversation":
        show_conversation_chat(helper, model, temperature, max_tokens)
    else:
        show_system_prompt_chat(helper, model, temperature, max_tokens)


def show_simple_chat(helper, model: str, temperature: float, max_tokens: int):
    """Show simple one-off chat interface."""

    st.subheader("🗨️ Simple Chat")
    st.markdown("Send a single message and get a response.")

    # Input area
    user_input = st.text_area("Your message:", placeholder="Type your message here...", height=100)

    col1, col2 = st.columns([1, 4])

    with col1:
        send_button = st.button("💬 Send", type="primary")

    with col2:
        if st.button("🧪 Try Example"):
            user_input = "Explain quantum computing in simple terms."
            st.rerun()

    if send_button and user_input.strip():
        with st.spinner(f"Getting response from {model}..."):
            try:
                response = helper.simple_chat(
                    model=model,
                    message=user_input,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )

                st.markdown("### 🤖 Response:")
                st.markdown(response)

                # Show some metadata
                with st.expander("📊 Response Details"):
                    st.write(f"**Model:** {model}")
                    st.write(f"**Temperature:** {temperature}")
                    st.write(f"**Max Tokens:** {max_tokens}")
                    st.write(f"**Response Length:** {len(response)} characters")

            except Exception as e:
                st.error(f"Error: {e}")


def show_conversation_chat(helper, model: str, temperature: float, max_tokens: int):
    """Show conversation chat with history."""

    st.subheader("🔄 Conversation Chat")
    st.markdown("Have a multi-turn conversation with memory.")

    # Initialize conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("### 📜 Conversation History")
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**{model}:** {message['content']}")

            if i < len(st.session_state.conversation_history) - 1:
                st.markdown("---")

    # Input area
    col1, col2 = st.columns([4, 1])

    with col1:
        user_input = st.text_input("Continue the conversation:", placeholder="Type your message...", key="conv_input")

    with col2:
        send_button = st.button("📤 Send", type="primary")
        if st.button("🗑️ Clear"):
            st.session_state.conversation_history = []
            st.rerun()

    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        with st.spinner(f"Getting response from {model}..."):
            try:
                response, updated_history = helper.chat_with_history(
                    model=model,
                    messages=st.session_state.conversation_history,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )

                st.session_state.conversation_history = updated_history
                st.rerun()

            except Exception as e:
                st.error(f"Error: {e}")
                # Remove the failed message
                st.session_state.conversation_history.pop()


def show_system_prompt_chat(helper, model: str, temperature: float, max_tokens: int):
    """Show chat with custom system prompts."""

    st.subheader("⚙️ System Prompt Chat")
    st.markdown("Define the AI's behavior with custom system prompts.")

    # System prompt examples
    system_examples = {
        "Default": "",
        "Helpful Assistant": "You are a helpful, harmless, and honest assistant.",
        "Code Expert": "You are an expert programmer who provides clear, concise code examples and explanations.",
        "Creative Writer": "You are a creative writer who tells engaging stories and uses vivid descriptions.",
        "Teacher": "You are a patient teacher who explains complex topics in simple terms with examples.",
        "Skeptical Analyst": "You are a critical thinker who questions assumptions and provides balanced analysis.",
    }

    col1, col2 = st.columns([3, 1])

    with col1:
        system_prompt = st.text_area(
            "System Prompt:",
            placeholder="Define how the AI should behave...",
            height=100,
            value=system_examples.get("Default", ""),
        )

    with col2:
        st.markdown("**Quick Examples:**")
        for name, prompt in system_examples.items():
            if st.button(f"📝 {name}", key=f"sys_{name}"):
                system_prompt = prompt
                st.rerun()

    # User message
    user_input = st.text_area("Your message:", placeholder="Type your message here...", height=100, key="sys_input")

    if st.button("🚀 Send with System Prompt", type="primary"):
        if user_input.strip():
            # Prepare messages with system prompt
            messages = []
            if system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_input})

            with st.spinner(f"Getting response from {model}..."):
                try:
                    response = helper.chat_with_messages(
                        model=model,
                        messages=messages,
                        options={"temperature": temperature, "num_predict": max_tokens},
                    )

                    st.markdown("### 🤖 Response:")
                    st.markdown(response.message.content)

                    # Show the full conversation
                    with st.expander("📋 Full Message Exchange"):
                        for msg in messages:
                            st.write(f"**{msg['role'].title()}:** {msg['content']}")
                        st.write(f"**Assistant:** {response.message.content}")

                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a message.")

    # Tips section
    with st.expander("💡 System Prompt Tips"):
        st.markdown(
            """
        **System prompts help you:**
        - Define the AI's personality and behavior
        - Set the context for the conversation
        - Specify the format of responses
        - Give the AI specific expertise or knowledge

        **Good practices:**
        - Be specific and clear
        - Include examples if needed
        - Keep it concise but complete
        - Test different variations
        """
        )


if __name__ == "__main__":
    show()
