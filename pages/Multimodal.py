"""
Multimodal Demo Page

Demonstrates multimodal (vision) functionality using OllamaHelper.
"""

import streamlit as st
import base64
from PIL import Image

from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_models_vision, add_select_temperature


def show():
    """Display the multimodal demo page."""
    
    st.title("🖼️ Multimodal Chat Demo")
    st.markdown("Chat with AI models about images and visual content.")
    
    # Get helper from session state
    helper = OllamaHelper()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Vision Settings")
        
        # Note: Only certain models support vision

        model = add_select_models_vision()
        temperature = add_select_temperature(default=0.7)
        
        analysis_type = st.radio(
            "Analysis Type",
            options=["General Description", "Detailed Analysis", "Creative Interpretation"],
            help="Choose the type of image analysis"
        )
        
        st.markdown("---")
        st.markdown("### 📷 Supported Formats")
        st.write("• JPEG, PNG, GIF")
        st.write("• Base64 encoded images")
        st.write("• URL references")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "🌐 Image URL", "🎨 Sample Images"])
    
    with tab1:
        show_image_upload(helper, model, temperature, analysis_type)
    
    with tab2:
        show_image_url(helper, model, temperature, analysis_type)
    
    with tab3:
        show_sample_images(helper, model, temperature, analysis_type)


def show_image_upload(helper, model: str, temperature: float, analysis_type: str):
    """Show image upload interface."""
    
    st.subheader("📤 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif'],
        help="Upload an image to analyze"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown("**Image Info:**")
            st.write(f"• Size: {image.size}")
            st.write(f"• Mode: {image.mode}")
            st.write(f"• Format: {uploaded_file.type}")
        
        with col2:
            # Analysis options
            custom_question = st.text_area(
                "Custom question about the image:",
                placeholder="What do you see in this image?",
                height=100
            )
            
            # Predefined questions based on analysis type
            if analysis_type == "General Description":
                default_question = "Describe what you see in this image in detail."
            elif analysis_type == "Detailed Analysis":
                default_question = "Provide a comprehensive analysis of this image, including objects, people, setting, colors, composition, and any notable details."
            else:  # Creative Interpretation
                default_question = "Interpret this image creatively. What story does it tell? What emotions does it evoke?"
            
            question = custom_question if custom_question.strip() else default_question
            
            st.markdown("**Question to ask:**")
            st.info(question)
            
            if st.button("🔍 Analyze Image", type="primary"):
                analyze_uploaded_image(helper, model, uploaded_file, question, temperature)


def show_image_url(helper, model: str, temperature: float, analysis_type: str):
    """Show image URL interface."""
    
    st.subheader("🌐 Analyze Image from URL")
    
    image_url = st.text_input(
        "Enter image URL:",
        placeholder="https://example.com/image.jpg",
        help="Enter a direct URL to an image file"
    )
    
    if image_url:
        try:
            # Try to display the image
            st.image(image_url, caption="Image from URL", use_column_width=True)
            
            # Analysis question
            custom_question = st.text_area(
                "Question about the image:",
                placeholder="What's happening in this image?",
                height=80
            )
            
            if analysis_type == "General Description":
                default_question = "What do you see in this image?"
            elif analysis_type == "Detailed Analysis":
                default_question = "Analyze this image in detail, describing all visible elements."
            else:
                default_question = "What story or meaning can you interpret from this image?"
            
            question = custom_question if custom_question.strip() else default_question
            
            if st.button("🔍 Analyze URL Image", type="primary"):
                analyze_image_url(helper, model, image_url, question, temperature)
                
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")


def show_sample_images(helper, model: str, temperature: float, analysis_type: str):
    """Show sample images for testing."""
    
    st.subheader("🎨 Sample Images for Testing")
    st.markdown("Use these sample images to test the vision capabilities.")
    
    # Sample image descriptions (you would replace these with actual sample images)
    samples = {
        "Nature Scene": {
            "description": "A beautiful landscape with mountains and lake",
            "prompt": "Describe the natural beauty and composition of this landscape."
        },
        "City Street": {
            "description": "Urban street scene with buildings and people",
            "prompt": "What can you tell me about this urban environment?"
        },
        "Art Piece": {
            "description": "A famous painting or artwork",
            "prompt": "Analyze the artistic elements, style, and composition."
        },
        "Food Photo": {
            "description": "Delicious meal or food preparation",
            "prompt": "Describe this food - what ingredients and cooking methods do you see?"
        },
        "Technology": {
            "description": "Modern technology or gadgets",
            "prompt": "What technology do you see and how might it be used?"
        }
    }
    
    st.markdown("**Sample Image Categories:**")
    
    col1, col2 = st.columns(2)
    
    for i, (category, info) in enumerate(samples.items()):
        with col1 if i % 2 == 0 else col2:
            with st.expander(f"📷 {category}"):
                st.write(info["description"])
                st.write(f"**Suggested prompt:** {info['prompt']}")
                
                if st.button(f"Use {category} Example", key=f"sample_{category}"):
                    st.info(f"Selected {category} - upload an image of this type to test!")
    
    # Instructions for using samples
    st.markdown("---")
    st.markdown("### 💡 How to use samples:")
    st.markdown("""
    1. **Choose a category** that interests you
    2. **Find or create an image** that fits the category
    3. **Upload the image** using the Upload tab
    4. **Use the suggested prompt** or create your own
    5. **Analyze** and see how the AI interprets visual content
    """)


def analyze_uploaded_image(helper, model: str, uploaded_file, question: str, temperature: float):
    """Analyze an uploaded image file."""
    
    with st.spinner(f"Analyzing image with {model}..."):
        try:
            # Convert image to base64
            image_bytes = uploaded_file.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode()
            
            # Use the multimodal chat function
            response = helper.multimodal_chat_base64(
                model=model,
                message=question,
                image_base64=image_base64,
                options={'temperature': temperature}
            )
            
            st.markdown("### 🤖 AI Analysis:")
            st.markdown(response)
            
            # Additional analysis options
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🎨 Creative Interpretation"):
                    creative_response = helper.multimodal_chat_base64(
                        model=model,
                        message="Give a creative, artistic interpretation of this image. What story does it tell?",
                        image_base64=image_base64,
                        options={'temperature': min(temperature + 0.3, 2.0)}
                    )
                    st.markdown("**Creative Interpretation:**")
                    st.markdown(creative_response)
            
            with col2:
                if st.button("🔍 Technical Details"):
                    technical_response = helper.multimodal_chat_base64(
                        model=model,
                        message="Describe the technical aspects of this image: lighting, composition, colors, camera angle, and photographic techniques.",
                        image_base64=image_base64,
                        options={'temperature': max(temperature - 0.2, 0.0)}
                    )
                    st.markdown("**Technical Analysis:**")
                    st.markdown(technical_response)
            
            with col3:
                if st.button("📊 Count Objects"):
                    count_response = helper.multimodal_chat_base64(
                        model=model,
                        message="Count and list all the distinct objects, people, or items you can identify in this image.",
                        image_base64=image_base64,
                        options={'temperature': 0.1}
                    )
                    st.markdown("**Object Count:**")
                    st.markdown(count_response)
            
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
            st.markdown("### 🔧 Troubleshooting:")
            st.markdown("""
            - Ensure the selected model supports vision (try 'gemma3' or 'llava')
            - Check that the image file is not corrupted
            - Try a smaller image size if the analysis fails
            - Make sure Ollama is running and accessible
            """)


def analyze_image_url(helper, model: str, image_url: str, question: str, temperature: float):
    """Analyze an image from URL."""
    
    with st.spinner(f"Analyzing image from URL with {model}..."):
        try:
            response = helper.multimodal_chat(
                model=model,
                message=question,
                image_path=image_url,  # Some models accept URLs directly
                options={'temperature': temperature}
            )
            
            st.markdown("### 🤖 AI Analysis:")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"Error analyzing image from URL: {e}")
            st.markdown("**Note:** Some models may not support direct URL analysis. Try downloading the image and uploading it instead.")


# Show tips and model info
def show_multimodal_tips():
    """Show tips for multimodal usage."""
    with st.expander("💡 Multimodal Tips"):
        st.markdown("""
        **Vision Model Capabilities:**
        - **Object Recognition** - Identify items, people, animals
        - **Scene Description** - Describe settings and environments
        - **Text Reading** - Read text within images (OCR)
        - **Color Analysis** - Identify and describe colors
        - **Composition** - Analyze artistic and photographic elements
        
        **Best Practices:**
        - Use clear, high-quality images
        - Be specific in your questions
        - Try different models for different tasks
        - Experiment with temperature settings
        
        **Model Recommendations:**
        - **LLaVA**: Excellent general vision capabilities
        - **Gemma3**: Good for basic image description
        - **BakLLaVA**: Alternative vision model option
        """)


if __name__ == "__main__":
    show()