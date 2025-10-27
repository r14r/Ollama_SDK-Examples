"""
Embeddings Demo Page

Demonstrates text embeddings functionality using OllamaHelper.
"""

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd

from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_models_embedding


def show():
    """Display the embeddings demo page."""
    
    st.title("🔗 Embeddings Demo")
    st.markdown("Explore text embeddings for similarity analysis and semantic search.")
    
    # Get helper from session state
    helper = OllamaHelper()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Embedding Settings")
        
        model = add_select_models_embedding()
        
        demo_mode = st.radio(
            "Demo Mode",
            options=["Text Similarity", "Semantic Search", "Clustering Visualization"],
            help="Choose embedding demonstration type"
        )
    
    # Main content area
    if demo_mode == "Text Similarity":
        show_text_similarity(helper, model)
    elif demo_mode == "Semantic Search":
        show_semantic_search(helper, model)
    else:
        show_clustering_visualization(helper, model)


def show_text_similarity(helper, model: str):
    """Show text similarity comparison."""
    
    st.subheader("📊 Text Similarity Analysis")
    st.markdown("Compare the semantic similarity between different texts.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text1 = st.text_area(
            "First Text:",
            placeholder="Enter first text to compare...",
            height=100,
            value="The cat sat on the mat."
        )
    
    with col2:
        text2 = st.text_area(
            "Second Text:",
            placeholder="Enter second text to compare...",
            height=100,
            value="A feline rested on the rug."
        )
    
    if st.button("🔍 Calculate Similarity", type="primary"):
        if text1.strip() and text2.strip():
            calculate_similarity(helper, model, text1, text2)
        else:
            st.warning("Please enter both texts.")
    
    # Quick examples
    st.markdown("---")
    st.markdown("### 🎯 Quick Examples")
    
    examples = [
        ("The weather is sunny today.", "It's a bright and clear day."),
        ("I love programming in Python.", "Python is my favorite coding language."),
        ("The quick brown fox jumps.", "A fast red dog runs."),
        ("Machine learning is fascinating.", "Cooking pasta is simple."),
    ]
    
    cols = st.columns(2)
    for i, (ex1, ex2) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Example {i+1}", key=f"sim_ex_{i}"):
                calculate_similarity(helper, model, ex1, ex2)


def show_semantic_search(helper, model: str):
    """Show semantic search demonstration."""
    
    st.subheader("🔎 Semantic Search")
    st.markdown("Search through a collection of texts using semantic similarity.")
    
    # Document collection
    if 'document_embeddings' not in st.session_state:
        st.session_state.document_embeddings = None
        st.session_state.documents = []
    
    # Add documents
    with st.expander("📚 Document Collection"):
        new_doc = st.text_area("Add a new document:", height=80)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add Document"):
                if new_doc.strip():
                    st.session_state.documents.append(new_doc.strip())
                    st.session_state.document_embeddings = None  # Reset embeddings
                    st.success("Document added!")
        
        with col2:
            if st.button("📄 Load Sample Documents"):
                sample_docs = [
                    "Artificial intelligence is transforming various industries.",
                    "Climate change requires immediate global action.",
                    "Python is a versatile programming language.",
                    "The stock market experienced significant volatility.",
                    "Renewable energy sources are becoming more efficient.",
                    "Machine learning algorithms can predict customer behavior.",
                    "Space exploration continues to push technological boundaries.",
                    "Healthy eating habits improve overall well-being."
                ]
                st.session_state.documents = sample_docs
                st.session_state.document_embeddings = None
                st.success("Sample documents loaded!")
        
        if st.session_state.documents:
            st.write(f"**Current documents ({len(st.session_state.documents)}):**")
            for i, doc in enumerate(st.session_state.documents):
                st.write(f"{i+1}. {doc[:100]}{'...' if len(doc) > 100 else ''}")
    
    # Search query
    if st.session_state.documents:
        query = st.text_input("Search query:", placeholder="Enter your search query...")
        
        if st.button("🔍 Search", type="primary") and query.strip():
            perform_semantic_search(helper, model, query)
    else:
        st.info("Add some documents to enable semantic search.")


def show_clustering_visualization(helper, model: str):
    """Show embedding clustering visualization."""
    
    st.subheader("📈 Embedding Clustering Visualization")
    st.markdown("Visualize how similar texts cluster together in embedding space.")
    
    # Text input for clustering
    texts_input = st.text_area(
        "Enter texts to visualize (one per line):",
        height=200,
        value="""Machine learning algorithms
Deep learning neural networks
Artificial intelligence systems
Python programming language
JavaScript web development
Data science analysis
Climate change effects
Global warming impact
Environmental protection
Renewable energy sources
Solar power generation
Wind energy turbines
Healthy eating habits
Exercise and fitness
Medical research studies
Scientific discoveries"""
    )
    
    if st.button("📊 Generate Visualization", type="primary"):
        texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
        if len(texts) >= 3:
            visualize_embeddings(helper, model, texts)
        else:
            st.warning("Please enter at least 3 texts for visualization.")


def calculate_similarity(helper, model: str, text1: str, text2: str):
    """Calculate and display similarity between two texts."""
    
    with st.spinner("Calculating embeddings..."):
        try:
            # Get embeddings for both texts
            embedding1 = helper.get_embeddings(model, text1)
            embedding2 = helper.get_embeddings(model, text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            # Display results
            st.markdown("### 📊 Similarity Results")
            
            # Similarity score with visual indicator
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Similarity Score", f"{similarity:.4f}")
                
                # Color-coded interpretation
                if similarity > 0.8:
                    st.success("Very Similar")
                elif similarity > 0.6:
                    st.info("Moderately Similar")
                elif similarity > 0.4:
                    st.warning("Somewhat Similar")
                else:
                    st.error("Not Similar")
            
            with col2:
                # Visual similarity bar
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = similarity,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Similarity"},
                    gauge = {
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.4], 'color': "lightgray"},
                            {'range': [0.4, 0.6], 'color': "yellow"},
                            {'range': [0.6, 0.8], 'color': "orange"},
                            {'range': [0.8, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.9
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # Embedding details
            with st.expander("🔍 Embedding Details"):
                st.write(f"**Text 1 embedding dimension:** {len(embedding1)}")
                st.write(f"**Text 2 embedding dimension:** {len(embedding2)}")
                st.write("**Cosine similarity formula:** cos(θ) = (A · B) / (||A|| ||B||)")
                
                # Show first few dimensions
                st.write("**First 10 dimensions of each embedding:**")
                df = pd.DataFrame({
                    'Dimension': range(1, 11),
                    'Text 1': embedding1[:10],
                    'Text 2': embedding2[:10]
                })
                st.dataframe(df)
        
        except Exception as e:
            st.error(f"Error calculating similarity: {e}")


def perform_semantic_search(helper, model: str, query: str):
    """Perform semantic search on document collection."""
    
    with st.spinner("Searching documents..."):
        try:
            # Get query embedding
            query_embedding = helper.get_embeddings(model, query)
            
            # Get or calculate document embeddings
            if st.session_state.document_embeddings is None:
                st.session_state.document_embeddings = []
                for doc in st.session_state.documents:
                    doc_embedding = helper.get_embeddings(model, doc)
                    st.session_state.document_embeddings.append(doc_embedding)
            
            # Calculate similarities
            similarities = []
            for doc_embedding in st.session_state.document_embeddings:
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append(similarity)
            
            # Sort results by similarity
            results = list(zip(st.session_state.documents, similarities))
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Display results
            st.markdown("### 🎯 Search Results")
            
            for i, (doc, similarity) in enumerate(results[:5]):  # Top 5 results
                with st.expander(f"Result {i+1} (Similarity: {similarity:.4f})"):
                    st.write(doc)
                    
                    # Similarity bar
                    progress_color = "green" if similarity > 0.7 else "orange" if similarity > 0.5 else "red"  # noqa: F841
                    st.progress(similarity)
            
            # Results visualization
            if len(results) > 1:
                df_results = pd.DataFrame({
                    'Document': [f"Doc {i+1}" for i in range(len(results))],
                    'Similarity': [sim for _, sim in results]
                })
                
                fig = px.bar(df_results, x='Document', y='Similarity',
                           title="Document Similarity Scores")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error performing search: {e}")


def visualize_embeddings(helper, model: str, texts: list):
    """Visualize embeddings using PCA dimensionality reduction."""
    
    with st.spinner("Generating embedding visualization..."):
        try:
            # Get embeddings for all texts
            embeddings = []
            for text in texts:
                embedding = helper.get_embeddings(model, text)
                embeddings.append(embedding)
            
            # Reduce dimensions using PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Create visualization
            df_viz = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'text': [text[:50] + '...' if len(text) > 50 else text for text in texts],
                'full_text': texts
            })
            
            fig = px.scatter(df_viz, x='x', y='y', hover_data=['full_text'],
                           title="Text Embeddings Visualization (PCA)")
            
            # Add text labels
            for i, row in df_viz.iterrows():
                fig.add_annotation(
                    x=row['x'], y=row['y'],
                    text=str(i+1),
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="white",
                    borderwidth=1
                )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show text index
            st.markdown("### 📋 Text Index")
            for i, text in enumerate(texts):
                st.write(f"**{i+1}.** {text}")
            
            # PCA explanation
            with st.expander("📊 About PCA Visualization"):
                st.markdown(f"""
                **Principal Component Analysis (PCA)** reduces the high-dimensional embedding space 
                (originally {len(embeddings[0])} dimensions) to 2 dimensions for visualization.
                
                - **Explained variance:** {pca.explained_variance_ratio_.sum():.2%}
                - **PC1 variance:** {pca.explained_variance_ratio_[0]:.2%}
                - **PC2 variance:** {pca.explained_variance_ratio_[1]:.2%}
                
                Points that are close together in this visualization have similar semantic meanings.
                """)
        
        except Exception as e:
            st.error(f"Error creating visualization: {e}")


if __name__ == "__main__":
    show()