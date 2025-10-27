"""
Text Generation Demo Page

Demonstrates text generation functionality using OllamaHelper.
"""

import streamlit as st

from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_max_tokens, add_select_models, add_select_temperature, add_select_top_p


def show():
    """Display the text generation demo page."""
    
    st.title("📝 Text Generation Demo")
    st.markdown("Generate text with various models and parameters.")
    
    # Get helper from session state
    helper = OllamaHelper()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Generation Settings")
        
        model = add_select_models()
        temperature = add_select_temperature()
        max_tokens = add_select_max_tokens()
        top_p = add_select_top_p()
        
        generation_mode = st.radio(
            "Generation Mode",
            options=["Simple Generation", "Creative Writing", "Code Generation"],
            help="Choose generation style"
        )
    
    # Main content area
    if generation_mode == "Simple Generation":
        show_simple_generation(helper, model, temperature, max_tokens, top_p)
    elif generation_mode == "Creative Writing":
        show_creative_writing(helper, model, temperature, max_tokens, top_p)
    else:
        show_code_generation(helper, model, temperature, max_tokens, top_p)


def show_simple_generation(helper, model: str, temperature: float, max_tokens: int, top_p: float):
    """Show simple text generation interface."""
    
    st.subheader("✏️ Simple Text Generation")
    st.markdown("Generate text based on a prompt.")
    
    # Input area
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Once upon a time...",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        generate_button = st.button("🚀 Generate", type="primary")
    
    with col2:
        if st.button("🎲 Random Prompt"):
            random_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology has advanced beyond our wildest dreams",
                "The secret to happiness lies in",
                "Climate change can be addressed by",
                "The most important lesson I learned was"
            ]
            import random
            prompt = random.choice(random_prompts)
            st.rerun()
    
    if generate_button and prompt.strip():
        with st.spinner(f"Generating text with {model}..."):
            try:
                response = helper.simple_generate(
                    model=model,
                    prompt=prompt,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                        'top_p': top_p
                    }
                )
                
                st.markdown("### 📄 Generated Text:")
                st.markdown(f"**Prompt:** {prompt}")
                st.markdown("**Generated:**")
                st.write(response)
                
                # Statistics
                with st.expander("📊 Generation Statistics"):
                    st.write(f"**Model:** {model}")
                    st.write(f"**Temperature:** {temperature}")
                    st.write(f"**Max Tokens:** {max_tokens}")
                    st.write(f"**Top P:** {top_p}")
                    st.write(f"**Generated Length:** {len(response)} characters")
                    st.write(f"**Word Count:** {len(response.split())} words")
                
            except Exception as e:
                st.error(f"Error: {e}")


def show_creative_writing(helper, model: str, temperature: float, max_tokens: int, top_p: float):
    """Show creative writing interface with templates."""
    
    st.subheader("🎨 Creative Writing")
    st.markdown("Generate creative content with writing templates.")
    
    # Writing templates
    templates = {
        "Story Beginning": "Write the opening paragraph of a story about",
        "Character Description": "Describe a character who is",
        "Setting Description": "Describe a mysterious place that is",
        "Dialogue": "Write a conversation between two people discussing",
        "Poetry": "Write a poem about",
        "Song Lyrics": "Write song lyrics about",
        "Haiku": "Write a haiku about",
        "Short Story": "Write a complete short story about"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_template = st.selectbox(
            "Choose Writing Template:",
            options=list(templates.keys())
        )
        
        topic = st.text_input(
            "Topic/Subject:",
            placeholder="a mysterious library, love, adventure, etc."
        )
        
        style_prompt = st.text_input(
            "Style (optional):",
            placeholder="in the style of Edgar Allan Poe, humorous, dramatic, etc."
        )
    
    with col2:
        st.markdown("**Template Examples:**")
        st.write("📖 Story Beginning")
        st.write("👤 Character Description") 
        st.write("🏞️ Setting Description")
        st.write("💬 Dialogue")
        st.write("🎵 Poetry & Songs")
        st.write("📚 Short Stories")
    
    # Build the full prompt
    if topic:
        base_prompt = f"{templates[selected_template]} {topic}"
        if style_prompt:
            full_prompt = f"{base_prompt}, {style_prompt}."
        else:
            full_prompt = f"{base_prompt}."
        
        st.markdown("**Full Prompt:**")
        st.info(full_prompt)
        
        if st.button("🎭 Generate Creative Content", type="primary"):
            with st.spinner(f"Creating with {model}..."):
                try:
                    response = helper.simple_generate(
                        model=model,
                        prompt=full_prompt,
                        options={
                            'temperature': min(temperature + 0.2, 2.0),  # Boost creativity
                            'num_predict': max_tokens,
                            'top_p': top_p
                        }
                    )
                    
                    st.markdown("### 🎨 Creative Output:")
                    st.markdown(response)
                    
                    # Rating system
                    st.markdown("### 📊 Rate this generation:")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        if st.button("⭐"):
                            st.write("Thanks for rating!")
                    with col2:
                        if st.button("⭐⭐"):
                            st.write("Thanks for rating!")
                    with col3:
                        if st.button("⭐⭐⭐"):
                            st.write("Thanks for rating!")
                    with col4:
                        if st.button("⭐⭐⭐⭐"):
                            st.write("Thanks for rating!")
                    with col5:
                        if st.button("⭐⭐⭐⭐⭐"):
                            st.write("Thanks for rating!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")


def show_code_generation(helper, model: str, temperature: float, max_tokens: int, top_p: float):
    """Show code generation interface."""
    
    st.subheader("💻 Code Generation")
    st.markdown("Generate code in various programming languages.")
    
    # Programming language selection
    languages = {
        "Python": "python",
        "JavaScript": "javascript",
        "Java": "java",
        "C++": "cpp",
        "C#": "csharp",
        "Go": "go",
        "Rust": "rust",
        "HTML/CSS": "html",
        "SQL": "sql"
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        language = st.selectbox(
            "Programming Language:",
            options=list(languages.keys())
        )
        
        task_description = st.text_area(
            "Describe what code you need:",
            placeholder="Create a function to calculate fibonacci numbers",
            height=100
        )
        
        include_comments = st.checkbox("Include comments", value=True)
        include_tests = st.checkbox("Include test examples", value=False)
    
    with col2:
        st.markdown("**Code Examples:**")
        code_examples = [
            "Sort an array",
            "Connect to database", 
            "Create REST API",
            "Parse JSON data",
            "File operations",
            "Web scraping",
            "Data visualization",
            "Algorithm implementation"
        ]
        
        for example in code_examples:
            if st.button(f"💡 {example}", key=f"code_{example}"):
                task_description = example
                st.rerun()
    
    if st.button("⚡ Generate Code", type="primary") and task_description.strip():
        # Build code generation prompt
        prompt = f"Write {language} code to {task_description}"
        
        if include_comments:
            prompt += ". Include clear comments explaining the code"
        
        if include_tests:
            prompt += ". Also provide test examples showing how to use the code"
        
        prompt += "."
        
        with st.spinner(f"Generating {language} code..."):
            try:
                response = helper.simple_generate(
                    model=model,
                    prompt=prompt,
                    options={
                        'temperature': max(temperature - 0.2, 0.0),  # Lower temp for code
                        'num_predict': max_tokens,
                        'top_p': top_p
                    }
                )
                
                st.markdown("### 💻 Generated Code:")
                st.code(response, language=languages[language])
                
                # Code actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Copy Code"):
                        st.write("Code copied to clipboard!")
                
                with col2:
                    if st.button("🔄 Regenerate"):
                        st.rerun()
                
                with col3:
                    if st.button("✨ Improve Code"):
                        improve_prompt = f"Improve this {language} code:\n\n{response}\n\nMake it more efficient and add error handling."
                        with st.spinner("Improving code..."):
                            improved = helper.simple_generate(model, improve_prompt)
                            st.markdown("### ✨ Improved Code:")
                            st.code(improved, language=languages[language])
                
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Code generation tips
    with st.expander("💡 Code Generation Tips"):
        st.markdown("""
        **For better code generation:**
        - Be specific about requirements
        - Mention performance constraints
        - Specify input/output formats
        - Request error handling
        - Ask for documentation
        
        **Model recommendations:**
        - **CodeLlama**: Best for code generation
        - **Llama3.1**: Good general coding
        - **Gemma3**: Decent for simple code
        """)


if __name__ == "__main__":
    show()