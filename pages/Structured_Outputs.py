"""
Structured Outputs Demo Page

Demonstrates structured output functionality using OllamaHelper.
"""

import streamlit as st
import json
from pydantic import BaseModel, Field
from typing import List, Optional


from lib.helper_ollama import OllamaHelper
from lib.helper_ui import add_select_models_tooling, add_select_temperature


def show():
    """Display the structured outputs demo page."""
    
    st.title("📊 Structured Outputs Demo")
    st.markdown("Generate structured data using JSON schemas and Pydantic models.")
    
    # Get helper from session state
    helper = OllamaHelper()
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Output Settings")

        model = add_select_models_tooling()
        temperature = add_select_temperature(default=0.3)
        
        output_format = st.radio(
            "Output Format",
            options=["Pydantic Models", "JSON Schema", "Business Data", "Creative Formats"],
            help="Choose structured output demonstration type"
        )
    
    # Main content area
    if output_format == "Pydantic Models":
        show_pydantic_models(helper, model, temperature)
    elif output_format == "JSON Schema":
        show_json_schema(helper, model, temperature)
    elif output_format == "Business Data":
        show_business_data(helper, model, temperature)
    else:
        show_creative_formats(helper, model, temperature)


def show_pydantic_models(helper, model: str, temperature: float):
    """Show Pydantic model structured outputs."""
    
    st.subheader("🏗️ Pydantic Models")
    st.markdown("Use Pydantic models to ensure type-safe, validated structured outputs.")
    
    # Define example Pydantic models
    class Person(BaseModel):
        name: str = Field(description="Full name of the person")
        age: int = Field(description="Age in years", ge=0, le=150)
        email: Optional[str] = Field(description="Email address", default=None)
        occupation: str = Field(description="Job title or profession")
        skills: List[str] = Field(description="List of skills or expertise")
    
    class Product(BaseModel):
        name: str = Field(description="Product name")
        price: float = Field(description="Price in USD", gt=0)
        category: str = Field(description="Product category")
        in_stock: bool = Field(description="Whether the product is in stock")
        rating: float = Field(description="Product rating", ge=0, le=5)
        features: List[str] = Field(description="Key product features")
    
    class BookReview(BaseModel):
        title: str = Field(description="Book title")
        author: str = Field(description="Book author")
        rating: int = Field(description="Rating from 1-5 stars", ge=1, le=5)
        review_text: str = Field(description="Review content")
        pros: List[str] = Field(description="Positive aspects")
        cons: List[str] = Field(description="Negative aspects")
        recommended: bool = Field(description="Whether to recommend the book")
    
    # Model selection
    model_options = {
        "Person Profile": Person,
        "Product Listing": Product,
        "Book Review": BookReview
    }
    
    selected_model_name = st.selectbox("Choose Pydantic Model:", list(model_options.keys()))
    selected_model = model_options[selected_model_name]
    
    # Show model schema
    with st.expander("🔍 Model Schema"):
        st.code(str(selected_model.model_json_schema()), language="json")
    
    # Example prompts for each model
    example_prompts = {
        "Person Profile": [
            "Create a profile for a software engineer named John",
            "Generate a person profile for a 25-year-old teacher",
            "Make up a profile for a freelance graphic designer"
        ],
        "Product Listing": [
            "Create a product listing for a wireless headphone",
            "Generate product info for a coffee maker",
            "Make a product entry for a smartphone"
        ],
        "Book Review": [
            "Write a review for '1984' by George Orwell",
            "Create a review for 'The Great Gatsby'",
            "Review 'To Kill a Mockingbird'"
        ]
    }
    
    st.markdown("### 🎯 Quick Examples:")
    for prompt in example_prompts[selected_model_name]:
        if st.button(f"📝 {prompt}", key=f"pyd_{prompt[:20]}"):
            generate_pydantic_output(helper, model, prompt, selected_model, temperature)
    
    # Custom prompt
    st.markdown("### 💬 Custom Prompt")
    custom_prompt = st.text_area(
        f"Enter prompt for {selected_model_name}:",
        placeholder=f"Generate a {selected_model_name.lower()} for..."
    )
    
    if st.button("🚀 Generate Structured Output", type="primary") and custom_prompt.strip():
        generate_pydantic_output(helper, model, custom_prompt, selected_model, temperature)


def show_json_schema(helper, model: str, temperature: float):
    """Show JSON schema structured outputs."""
    
    st.subheader("📋 JSON Schema")
    st.markdown("Define custom JSON schemas for precise output formatting.")
    
    # Predefined schemas
    schemas = {
        "Contact Information": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "phone": {"type": "string", "description": "Phone number"},
                "email": {"type": "string", "description": "Email address"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                        "country": {"type": "string"}
                    },
                    "required": ["street", "city", "country"]
                }
            },
            "required": ["name", "phone", "email", "address"]
        },
        "Recipe": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Recipe name"},
                "prep_time": {"type": "integer", "description": "Preparation time in minutes"},
                "cook_time": {"type": "integer", "description": "Cooking time in minutes"},
                "servings": {"type": "integer", "description": "Number of servings"},
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "amount": {"type": "string"},
                            "unit": {"type": "string"}
                        }
                    }
                },
                "instructions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}
            },
            "required": ["name", "prep_time", "cook_time", "servings", "ingredients", "instructions"]
        },
        "Event Planning": {
            "type": "object",
            "properties": {
                "event_name": {"type": "string"},
                "date": {"type": "string", "format": "date"},
                "location": {"type": "string"},
                "attendees": {"type": "integer", "minimum": 1},
                "budget": {"type": "number", "minimum": 0},
                "activities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "duration": {"type": "integer", "description": "Duration in minutes"},
                            "cost": {"type": "number", "minimum": 0}
                        }
                    }
                },
                "requirements": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
    
    # Schema selection
    selected_schema_name = st.selectbox("Choose JSON Schema:", list(schemas.keys()))
    selected_schema = schemas[selected_schema_name]
    
    # Show schema
    with st.expander("🔍 JSON Schema"):
        st.code(json.dumps(selected_schema, indent=2), language="json")
    
    # Example prompts
    schema_prompts = {
        "Contact Information": [
            "Create contact info for a business owner",
            "Generate contact details for a freelancer",
            "Make contact info for a restaurant"
        ],
        "Recipe": [
            "Create a recipe for chocolate chip cookies",
            "Generate a pasta recipe",
            "Make a healthy salad recipe"
        ],
        "Event Planning": [
            "Plan a birthday party for 20 people",
            "Create a corporate team building event",
            "Plan a wedding reception"
        ]
    }
    
    st.markdown("### 🎯 Schema Examples:")
    for prompt in schema_prompts[selected_schema_name]:
        if st.button(f"📝 {prompt}", key=f"json_{prompt[:20]}"):
            generate_json_schema_output(helper, model, prompt, selected_schema, temperature)
    
    # Custom schema editor
    st.markdown("### 🛠️ Custom Schema")
    with st.expander("Create Custom Schema"):
        custom_schema_text = st.text_area(
            "Define your JSON schema:",
            value=json.dumps(selected_schema, indent=2),
            height=200
        )
        
        custom_prompt = st.text_input("Prompt for custom schema:")
        
        if st.button("🚀 Generate with Custom Schema"):
            try:
                custom_schema = json.loads(custom_schema_text)
                generate_json_schema_output(helper, model, custom_prompt, custom_schema, temperature)
            except json.JSONDecodeError:
                st.error("Invalid JSON schema format")


def show_business_data(helper, model: str, temperature: float):
    """Show business-focused structured outputs."""
    
    st.subheader("💼 Business Data Formats")
    st.markdown("Generate structured business data for real-world applications.")
    
    # Business data models
    class Invoice(BaseModel):
        invoice_number: str = Field(description="Unique invoice identifier")
        date: str = Field(description="Invoice date in YYYY-MM-DD format")
        customer_name: str = Field(description="Customer name")
        customer_email: str = Field(description="Customer email")
        items: List[dict] = Field(description="List of invoice items")
        subtotal: float = Field(description="Subtotal amount")
        tax: float = Field(description="Tax amount")
        total: float = Field(description="Total amount")
        payment_status: str = Field(description="Payment status", pattern="^(paid|pending|overdue)$")
    
    class Employee(BaseModel):
        employee_id: str = Field(description="Unique employee ID")
        name: str = Field(description="Full name")
        department: str = Field(description="Department name")
        position: str = Field(description="Job position")
        salary: float = Field(description="Annual salary", gt=0)
        start_date: str = Field(description="Start date in YYYY-MM-DD format")
        skills: List[str] = Field(description="Professional skills")
        performance_rating: float = Field(description="Performance rating 1-5", ge=1, le=5)
    
    class ProjectPlan(BaseModel):
        project_name: str = Field(description="Project name")
        description: str = Field(description="Project description")
        start_date: str = Field(description="Start date")
        end_date: str = Field(description="End date")
        budget: float = Field(description="Project budget")
        team_members: List[str] = Field(description="Team member names")
        milestones: List[dict] = Field(description="Project milestones")
        risks: List[str] = Field(description="Identified risks")
        success_criteria: List[str] = Field(description="Success criteria")
    
    business_models = {
        "Invoice": Invoice,
        "Employee Record": Employee,
        "Project Plan": ProjectPlan
    }
    
    selected_business_model = st.selectbox("Choose Business Data Type:", list(business_models.keys()))
    model_class = business_models[selected_business_model]
    
    # Business scenarios
    business_scenarios = {
        "Invoice": [
            "Create an invoice for web design services",
            "Generate an invoice for consulting work",
            "Make an invoice for software development"
        ],
        "Employee Record": [
            "Create an employee record for a new software developer",
            "Generate employee data for a marketing manager",
            "Make an employee record for a data scientist"
        ],
        "Project Plan": [
            "Plan a website redesign project",
            "Create a mobile app development plan",
            "Plan a marketing campaign project"
        ]
    }
    
    st.markdown("### 🎯 Business Scenarios:")
    for scenario in business_scenarios[selected_business_model]:
        if st.button(f"💼 {scenario}", key=f"biz_{scenario[:20]}"):
            generate_pydantic_output(helper, model, scenario, model_class, temperature)
    
    # Custom business prompt
    st.markdown("### 💬 Custom Business Request")
    business_prompt = st.text_area(
        f"Describe the {selected_business_model.lower()} you need:",
        placeholder=f"Create a {selected_business_model.lower()} for..."
    )
    
    if st.button("📊 Generate Business Data", type="primary") and business_prompt.strip():
        generate_pydantic_output(helper, model, business_prompt, model_class, temperature)


def show_creative_formats(helper, model: str, temperature: float):
    """Show creative structured output formats."""
    
    st.subheader("🎨 Creative Formats")
    st.markdown("Generate creative content in structured formats.")
    
    # Creative models
    class Story(BaseModel):
        title: str = Field(description="Story title")
        genre: str = Field(description="Story genre")
        characters: List[dict] = Field(description="Main characters with names and descriptions")
        setting: str = Field(description="Story setting")
        plot_summary: str = Field(description="Brief plot summary")
        chapters: List[dict] = Field(description="Chapter outlines")
        themes: List[str] = Field(description="Story themes")
        target_audience: str = Field(description="Target audience")
    
    class SongLyrics(BaseModel):
        title: str = Field(description="Song title")
        artist: str = Field(description="Artist name")
        genre: str = Field(description="Music genre")
        verses: List[str] = Field(description="Song verses")
        chorus: str = Field(description="Chorus lyrics")
        bridge: Optional[str] = Field(description="Bridge lyrics")
        mood: str = Field(description="Song mood")
        instruments: List[str] = Field(description="Suggested instruments")
    
    class GameConcept(BaseModel):
        name: str = Field(description="Game name")
        genre: str = Field(description="Game genre")
        platform: List[str] = Field(description="Target platforms")
        storyline: str = Field(description="Game storyline")
        characters: List[dict] = Field(description="Main characters")
        gameplay_mechanics: List[str] = Field(description="Core gameplay mechanics")
        levels: List[dict] = Field(description="Level descriptions")
        target_rating: str = Field(description="ESRB rating")
    
    creative_models = {
        "Story Outline": Story,
        "Song Lyrics": SongLyrics,
        "Game Concept": GameConcept
    }
    
    selected_creative_model = st.selectbox("Choose Creative Format:", list(creative_models.keys()))
    creative_class = creative_models[selected_creative_model]
    
    # Creative prompts
    creative_prompts = {
        "Story Outline": [
            "Create a sci-fi adventure story",
            "Generate a mystery thriller outline",
            "Make a fantasy romance story"
        ],
        "Song Lyrics": [
            "Write a pop song about friendship",
            "Create folk song lyrics about nature",
            "Generate a rock song about perseverance"
        ],
        "Game Concept": [
            "Design a puzzle platformer game",
            "Create an RPG adventure concept",
            "Generate a strategy game idea"
        ]
    }
    
    st.markdown("### 🎯 Creative Ideas:")
    for prompt in creative_prompts[selected_creative_model]:
        if st.button(f"🎨 {prompt}", key=f"creative_{prompt[:20]}"):
            generate_pydantic_output(helper, model, prompt, creative_class, temperature + 0.2)  # Higher temp for creativity
    
    # Custom creative prompt
    st.markdown("### 💫 Your Creative Vision")
    creative_prompt = st.text_area(
        f"Describe your {selected_creative_model.lower()} idea:",
        placeholder=f"Create a {selected_creative_model.lower()} about..."
    )
    
    if st.button("✨ Generate Creative Content", type="primary") and creative_prompt.strip():
        generate_pydantic_output(helper, model, creative_prompt, creative_class, min(temperature + 0.3, 1.0))


def generate_pydantic_output(helper, model: str, prompt: str, pydantic_model: BaseModel, temperature: float):
    """Generate structured output using Pydantic model."""
    
    st.markdown(f"### 🎯 Generating: {pydantic_model.__name__}")
    
    with st.spinner("Generating structured output..."):
        try:
            result = helper.chat_with_structured_output(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                schema=pydantic_model,
                options={'temperature': temperature}
            )
            
            st.success("✅ Structured output generated successfully!")
            
            # Display the result
            st.markdown("**Generated Data:**")
            st.json(result.dict() if hasattr(result, 'dict') else result)
            
            # Validation info
            with st.expander("✅ Validation Details"):
                st.write(f"**Model:** {pydantic_model.__name__}")
                st.write(f"**Fields:** {len(pydantic_model.__fields__)}")
                st.write("**Validation:** All fields passed validation ✅")
                
                # Show field types
                st.markdown("**Field Types:**")
                for field_name, field_info in pydantic_model.__fields__.items():
                    st.write(f"• `{field_name}`: {field_info.type_}")
        
        except Exception as e:
            st.error(f"Error generating structured output: {e}")
            st.markdown("**Troubleshooting:**")
            st.markdown("- Try a simpler prompt")
            st.markdown("- Lower the temperature")
            st.markdown("- Ensure the model supports structured outputs")


def generate_json_schema_output(helper, model: str, prompt: str, schema: dict, temperature: float):
    """Generate structured output using JSON schema."""
    
    st.markdown("### 🎯 Generating JSON Schema Output")
    
    with st.spinner("Generating structured output..."):
        try:
            result = helper.chat_with_structured_output(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                schema=schema,
                options={'temperature': temperature}
            )
            
            st.success("✅ JSON schema output generated successfully!")
            
            # Display the result
            st.markdown("**Generated Data:**")
            st.json(result)
            
            # Schema validation info
            with st.expander("📋 Schema Validation"):
                st.write("**Schema:** Valid JSON structure ✅")
                st.write(f"**Properties:** {len(schema.get('properties', {}))}")
                if 'required' in schema:
                    st.write(f"**Required Fields:** {', '.join(schema['required'])}")
        
        except Exception as e:
            st.error(f"Error generating JSON schema output: {e}")


if __name__ == "__main__":
    show()