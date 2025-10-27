import streamlit as st
import time

from lib.helper_ollama import helper


def show():
    """Display the model management demo page."""
    
    st.title("⚙️ Model Management Demo")
    st.markdown("Manage Ollama models: list, pull, show details, and monitor running models.")
        
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 List Models", "⬇️ Pull Models", "🔍 Model Details", "🖥️ Running Models"])
    
    with tab1:
        show_list_models(helper)
    
    with tab2:
        show_pull_models(helper)
    
    with tab3:
        show_model_details(helper)
    
    with tab4:
        show_running_models(helper)


def show_list_models(helper):
    """Show available models interface."""
    
    st.subheader("📋 Available Models")
    st.markdown("View all models available in your Ollama installation.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Refresh Models", type="primary"):
            st.rerun()
        
        auto_refresh = st.checkbox("Auto-refresh every 30s")
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    with col2:
        try:
            models = helper.models_list(with_details=True)
            st.success(f"Found {len(models)} models")
        except Exception as e:
            st.error(f"Error fetching models: {e}")
            return
    
    if models:
        # Display models in a nice format
        for i, model in enumerate(models):
            with st.expander(f"🤖 {model.model}", expanded=i < 3):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {model.model}")
                    size_mb = model.size.real / (1024 * 1024)
                    size_gb = size_mb / 1024
                    
                    if size_gb >= 1:
                        st.write(f"**Size:** {size_gb:.2f} GB")
                    else:
                        st.write(f"**Size:** {size_mb:.2f} MB")
                    
                    st.write(f"**Modified:** {model.modified_at}")
                
                with col2:
                    if model.details:
                        st.write(f"**Format:** {model.details.format}")
                        st.write(f"**Family:** {model.details.family}")
                        st.write(f"**Parameters:** {model.details.parameter_size}")
                        st.write(f"**Quantization:** {model.details.quantization_level}")
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"🧪 Test {model.model}", key=f"test_{i}"):
                        test_model(helper, model.model)
                
                with col2:
                    if st.button("📊 Details", key=f"details_{i}"):
                        show_detailed_model_info(helper, model.model)
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.warning(f"Model removal for {model.model} would require direct Ollama CLI")
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📊 Model Statistics")
        
        total_size = sum(model.size.real for model in models)
        total_gb = total_size / (1024 * 1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", len(models))
        with col2:
            st.metric("Total Size", f"{total_gb:.2f} GB")
        with col3:
            avg_size = total_gb / len(models) if models else 0
            st.metric("Average Size", f"{avg_size:.2f} GB")
    
    else:
        st.info("No models found. Use the 'Pull Models' tab to download some models.")


def show_pull_models(helper):
    """Show model pulling interface."""
    
    st.subheader("⬇️ Pull New Models")
    st.markdown("Download models from the Ollama library.")
    
    # Popular models
    popular_models = {
        "Text Models": [
            "llama3.1:8b", "llama3.1:70b", "llama3.2:3b", "llama3.2:1b",
            "gemma3:2b", "gemma3:9b", "gemma3:27b",
            "mistral:7b", "mixtral:8x7b", 
            "codellama:7b", "codellama:13b"
        ],
        "Vision Models": [
            "llava:7b", "llava:13b", "llava:34b",
            "bakllava:7b"
        ],
        "Embedding Models": [
            "nomic-embed-text", "all-minilm:l6-v2", "all-minilm:l12-v2"
        ],
        "Code Models": [
            "codellama:7b-code", "codellama:13b-code",
            "codegemma:2b", "codegemma:7b"
        ]
    }
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        category = st.selectbox("Model Category:", list(popular_models.keys()))
        model_name = st.selectbox("Select Model:", popular_models[category])
        
        # Custom model input
        st.markdown("**Or enter custom model:**")
        custom_model = st.text_input("Custom model name:", placeholder="username/modelname:tag")
        
        model_to_pull = custom_model if custom_model.strip() else model_name
    
    with col2:
        st.markdown("**Model Information:**")
        if model_to_pull:
            st.info(f"Will pull: `{model_to_pull}`")
            
            # Estimated info based on model name
            if "3b" in model_to_pull.lower() or "2b" in model_to_pull.lower():
                st.write("📏 **Size:** ~2-4 GB")
                st.write("⚡ **Speed:** Very Fast")
                st.write("🧠 **Quality:** Good")
            elif "7b" in model_to_pull.lower():
                st.write("📏 **Size:** ~4-8 GB")
                st.write("⚡ **Speed:** Fast")
                st.write("🧠 **Quality:** Very Good")
            elif "13b" in model_to_pull.lower():
                st.write("📏 **Size:** ~8-15 GB")
                st.write("⚡ **Speed:** Medium")
                st.write("🧠 **Quality:** Excellent")
            elif "70b" in model_to_pull.lower():
                st.write("📏 **Size:** ~40-80 GB")
                st.write("⚡ **Speed:** Slow")
                st.write("🧠 **Quality:** Outstanding")
    
    # Pull options
    with_progress = st.checkbox("Show detailed progress", value=True)
    
    # Pull button
    if st.button("⬇️ Pull Model", type="primary"):
        if model_to_pull:
            pull_model_with_progress(helper, model_to_pull, with_progress)
        else:
            st.warning("Please select or enter a model name.")
    
    # Pull status
    if 'pulling_model' in st.session_state and st.session_state.pulling_model:
        st.info(f"Currently pulling: {st.session_state.pulling_model}")


def show_model_details(helper):
    """Show detailed model information interface."""
    
    st.subheader("🔍 Model Details")
    st.markdown("Get comprehensive information about a specific model.")
    
    # Get available models for selection
    try:
        models = helper.models_list(with_details=True)
        model_names = [model.model for model in models]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return
    
    if not model_names:
        st.warning("No models available. Pull some models first.")
        return
    
    selected_model = st.selectbox("Select model to inspect:", model_names)
    
    if st.button("🔍 Get Model Details", type="primary"):
        show_detailed_model_info(helper, selected_model)


def show_running_models(helper):
    """Show currently running models interface."""
    
    st.subheader("🖥️ Running Models")
    st.markdown("Monitor currently loaded models and their resource usage.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Refresh", type="primary"):
            st.rerun()
        
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    try:
        running_models = helper.list_running_models()
        
        if running_models:
            st.success(f"Found {len(running_models)} running models")
            
            for i, model in enumerate(running_models):
                with st.expander(f"🔄 {model.model} (Running)", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Model:** {model.model}")
                        st.write(f"**Digest:** {model.digest[:16]}...")
                        st.write(f"**Expires:** {model.expires_at}")
                    
                    with col2:
                        size_mb = model.size / (1024 * 1024) if model.size else 0
                        vram_mb = model.size_vram / (1024 * 1024) if model.size_vram else 0
                        
                        st.write(f"**Memory:** {size_mb:.1f} MB")
                        st.write(f"**VRAM:** {vram_mb:.1f} MB")
                        st.write(f"**Context Length:** {model.context_length}")
                    
                    # Model actions
                    if st.button("🧪 Test Model", key=f"test_running_{i}"):
                        test_model(helper, model.model)
            
            # Resource usage summary
            total_memory = sum(model.size for model in running_models if model.size)
            total_vram = sum(model.size_vram for model in running_models if model.size_vram)
            
            st.markdown("---")
            st.markdown("### 📊 Resource Usage")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Running Models", len(running_models))
            with col2:
                st.metric("Total Memory", f"{total_memory / (1024*1024):.1f} MB")
            with col3:
                st.metric("Total VRAM", f"{total_vram / (1024*1024):.1f} MB")
        
        else:
            st.info("No models are currently running.")
            st.markdown("**To load a model:** Send a chat request or generation request to any model.")
    
    except Exception as e:
        st.error(f"Error fetching running models: {e}")
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()


def test_model(helper, model_name: str):
    """Test a model with a simple query."""
    
    with st.spinner(f"Testing {model_name}..."):
        try:
            response = helper.simple_chat(model_name, "Hello! Please respond with a short greeting.")
            st.success(f"✅ Model {model_name} is working!")
            st.info(f"**Response:** {response}")
        except Exception as e:
            st.error(f"❌ Model {model_name} test failed: {e}")


def show_detailed_model_info(helper, model_name: str):
    """Show comprehensive model information."""
    
    with st.spinner(f"Fetching details for {model_name}..."):
        try:
            model_info = helper.get_model_info(model_name)
            
            st.markdown(f"### 📊 {model_name} Details")
            
            # Basic info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information:**")
                st.write(f"**Modified:** {model_info.modified_at}")
                if model_info.details:
                    st.write(f"**Format:** {model_info.details.format}")
                    st.write(f"**Family:** {model_info.details.family}")
                    st.write(f"**Parameter Size:** {model_info.details.parameter_size}")
                    st.write(f"**Quantization:** {model_info.details.quantization_level}")
            
            with col2:
                st.markdown("**Capabilities:**")
                if model_info.capabilities:
                    for capability in model_info.capabilities:
                        st.write(f"✅ {capability}")
                else:
                    st.write("No specific capabilities listed")
            
            # Template and modelfile
            if model_info.template:
                with st.expander("📝 Chat Template"):
                    st.code(model_info.template, language="text")
            
            if model_info.modelfile:
                with st.expander("🔧 Modelfile"):
                    st.code(model_info.modelfile, language="dockerfile")
            
            # Parameters
            if model_info.parameters:
                with st.expander("⚙️ Parameters"):
                    st.code(model_info.parameters, language="json")
            
            # Model info
            if model_info.modelinfo:
                with st.expander("ℹ️ Model Information"):
                    st.code(str(model_info.modelinfo), language="json")
            
            # License
            if model_info.license:
                with st.expander("📄 License"):
                    st.markdown(model_info.license)
        
        except Exception as e:
            st.error(f"Error fetching model details: {e}")


def pull_model_with_progress(helper, model_name: str, show_progress: bool):
    """Pull a model with optional progress display."""
    
    st.session_state.pulling_model = model_name
    
    if show_progress:
        # Use progress bar for detailed progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            total_size = None
            downloaded = 0
            
            for progress in helper.pull_model(model_name, stream=True):
                status = progress.get('status', '')
                
                if 'total' in progress:
                    total_size = progress['total']
                
                if 'completed' in progress and total_size:
                    downloaded = progress['completed']
                    percentage = downloaded / total_size
                    progress_bar.progress(percentage)
                    status_text.text(f"{status} ({percentage:.1%})")
                else:
                    status_text.text(status)
            
            st.success(f"✅ Successfully pulled {model_name}")
            
        except Exception as e:
            st.error(f"❌ Error pulling {model_name}: {e}")
    
    else:
        # Simple pull without detailed progress
        with st.spinner(f"Pulling {model_name}..."):
            try:
                helper.pull_model(model_name)
                st.success(f"✅ Successfully pulled {model_name}")
            except Exception as e:
                st.error(f"❌ Error pulling {model_name}: {e}")
    
    st.session_state.pulling_model = None


if __name__ == "__main__":
    show()