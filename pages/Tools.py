import math
from datetime import datetime
from typing import Any

import streamlit as st
from lib.helper_ollama import helper


def show():
    """Display the tools demo page."""

    st.title("🛠️ Tool Calling Demo")
    st.markdown("Explore AI function calling with custom tools and real-world examples.")

    # Sidebar controls
    with st.sidebar:
        st.subheader("🔧 Tool Settings")

        model = st.selectbox(
            "Select Model",
            options=helper.models_list(with_details=False),
            help="Choose a model that supports tool calling",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Lower temperature for more reliable tool calling",
        )

        demo_mode = st.radio(
            "Demo Mode",
            options=["Math Tools", "Data Tools", "Web Tools", "Custom Tools"],
            help="Choose tool demonstration type",
        )

    # Main content area
    if demo_mode == "Math Tools":
        show_math_tools(helper, model, temperature)
    elif demo_mode == "Data Tools":
        show_data_tools(helper, model, temperature)
    elif demo_mode == "Web Tools":
        show_web_tools(helper, model, temperature)
    else:
        show_custom_tools(helper, model, temperature)


def show_math_tools(helper, model: str, temperature: float):
    """Show mathematical tool calling demo."""

    st.subheader("🔢 Mathematical Tools")
    st.markdown("AI can call mathematical functions to perform calculations.")

    # Define math tools
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    def calculate_area_circle(radius: float) -> float:
        """Calculate the area of a circle given its radius."""
        return math.pi * radius**2

    def calculate_factorial(n: int) -> int:
        """Calculate factorial of a number."""
        if n < 0:
            raise ValueError("Cannot calculate factorial of negative number")

        return int(math.factorial(n))

    # Available functions
    available_functions = {
        "add_numbers": add_numbers,
        "multiply_numbers": multiply_numbers,
        "calculate_area_circle": calculate_area_circle,
        "calculate_factorial": calculate_factorial,
    }

    # Tool definitions
    tools = [add_numbers, multiply_numbers, calculate_area_circle, calculate_factorial]

    # Example queries
    st.markdown("### 🎯 Try These Examples:")
    example_queries = [
        "What is 15 + 27?",
        "Calculate 8 times 12",
        "What's the area of a circle with radius 5?",
        "Calculate factorial of 6",
        "What is (10 + 5) * 3?",
        "Find the area of a circle with radius 7.5 and add 10 to it",
    ]

    cols = st.columns(2)
    for i, query in enumerate(example_queries):
        with cols[i % 2]:
            if st.button(f"📝 {query}", key=f"math_ex_{i}"):
                execute_tool_query(helper, model, query, tools, available_functions, temperature)

    # Custom query
    st.markdown("### 💬 Ask Your Own Math Question")
    user_query = st.text_input(
        "Enter a mathematical question:",
        placeholder="Calculate the area of a circle with radius 10 and multiply by 2",
    )

    if st.button("🚀 Execute", type="primary") and user_query.strip():
        execute_tool_query(helper, model, user_query, tools, available_functions, temperature)


def show_data_tools(helper, model: str, temperature: float):
    """Show data processing tool calling demo."""

    st.subheader("📊 Data Processing Tools")
    st.markdown("AI can call functions to process and analyze data.")

    # Define data tools
    def analyze_list(numbers: list) -> dict:
        """Analyze a list of numbers and return statistics."""
        if not numbers:
            return {"error": "Empty list provided"}

        return {
            "count": len(numbers),
            "sum": sum(numbers),
            "average": sum(numbers) / len(numbers),
            "min": min(numbers),
            "max": max(numbers),
            "range": max(numbers) - min(numbers),
        }

    def filter_list(numbers: list, threshold: float, operator: str = "greater") -> list:
        """Filter a list of numbers based on a threshold."""
        if operator == "greater":
            return [n for n in numbers if n > threshold]
        elif operator == "less":
            return [n for n in numbers if n < threshold]
        elif operator == "equal":
            return [n for n in numbers if n == threshold]
        else:
            return numbers

    def sort_data(data: list, reverse: bool = False) -> list:
        """Sort a list of data."""
        return sorted(data, reverse=reverse)

    def calculate_percentage(part: float, total: float) -> float:
        """Calculate percentage of part relative to total."""
        if total == 0:
            return 0
        return (part / total) * 100

    # Available functions
    available_functions = {
        "analyze_list": analyze_list,
        "filter_list": filter_list,
        "sort_data": sort_data,
        "calculate_percentage": calculate_percentage,
    }

    # Tool definitions (simplified for demo)
    tools = [analyze_list, filter_list, sort_data, calculate_percentage]

    # Sample data
    sample_data = [12, 45, 23, 67, 34, 89, 15, 78, 56, 41]
    st.markdown(f"**Sample Data:** {sample_data}")

    # Example queries
    st.markdown("### 🎯 Data Analysis Examples:")
    data_queries = [
        f"Analyze this list of numbers: {sample_data}",
        f"Filter numbers greater than 50 from: {sample_data}",
        f"Sort this data in descending order: {sample_data}",
        "What percentage is 25 out of 200?",
        f"Find the average of {sample_data} and tell me how many are above average",
    ]

    for i, query in enumerate(data_queries):
        if st.button(f"📊 {query[:50]}...", key=f"data_ex_{i}"):
            execute_tool_query(helper, model, query, tools, available_functions, temperature)

    # Custom data query
    st.markdown("### 💬 Custom Data Question")
    custom_data = st.text_input("Enter comma-separated numbers:", placeholder="1,2,3,4,5")
    data_query = st.text_input(
        "What would you like to do with this data?",
        placeholder="Find the average and maximum value",
    )

    if st.button("📈 Analyze Data", type="primary") and custom_data and data_query:
        try:
            numbers = [float(x.strip()) for x in custom_data.split(",")]
            full_query = f"{data_query} for this data: {numbers}"
            execute_tool_query(helper, model, full_query, tools, available_functions, temperature)
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")


def show_web_tools(helper, model: str, temperature: float):
    """Show web-related tool calling demo."""

    st.subheader("🌐 Web Tools")
    st.markdown("AI can call functions to interact with web services and APIs.")

    # Define web tools
    def get_current_time(timezone: str = "UTC") -> str:
        """Get current time in specified timezone."""
        now = datetime.now()
        return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"

    def calculate_days_between(date1: str, date2: str) -> int:
        """Calculate days between two dates (YYYY-MM-DD format)."""
        try:
            d1 = datetime.strptime(date1, "%Y-%m-%d")
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            return abs((d2 - d1).days)
        except ValueError:
            # Return -1 to indicate invalid input while keeping the return type int
            return -1

    def generate_password(length: int = 12, include_special: bool = True) -> str:
        """Generate a random password."""
        import random
        import string

        characters = string.ascii_letters + string.digits
        if include_special:
            characters += "!@#$%^&*"

        return "".join(random.choice(characters) for _ in range(length))

    def url_shortener_info(url: str) -> dict:
        """Get information about a URL (mock function)."""
        return {
            "original_url": url,
            "domain": url.split("/")[2] if "//" in url else "unknown",
            "protocol": url.split("://")[0] if "://" in url else "unknown",
            "length": len(url),
            "is_secure": url.startswith("https://"),
        }

    # Available functions
    available_functions = {
        "get_current_time": get_current_time,
        "calculate_days_between": calculate_days_between,
        "generate_password": generate_password,
        "url_shortener_info": url_shortener_info,
    }

    tools = [get_current_time, calculate_days_between, generate_password, url_shortener_info]

    # Example queries
    st.markdown("### 🎯 Web Tool Examples:")
    web_queries = [
        "What time is it now?",
        "How many days between 2024-01-01 and 2024-12-31?",
        "Generate a secure password with 16 characters",
        "Analyze this URL: https://www.example.com/page",
        "Generate a password and tell me about the URL https://github.com",
    ]

    for i, query in enumerate(web_queries):
        if st.button(f"🌐 {query}", key=f"web_ex_{i}"):
            execute_tool_query(helper, model, query, tools, available_functions, temperature)

    # Custom web query
    st.markdown("### 💬 Custom Web Question")
    web_query = st.text_area(
        "Ask about time, dates, passwords, or URLs:",
        placeholder="Generate a 20-character password and tell me how many days until Christmas",
    )

    if st.button("🚀 Execute Web Tools", type="primary") and web_query.strip():
        execute_tool_query(helper, model, web_query, tools, available_functions, temperature)


def show_custom_tools(helper, model: str, temperature: float):
    """Show custom tool creation and execution."""

    st.subheader("🔧 Custom Tools")
    st.markdown("Create and test your own custom tools.")

    # Tool definition interface
    st.markdown("### 🛠️ Define Your Tool")

    col1, col2 = st.columns(2)

    with col1:
        tool_name = st.text_input("Tool Name:", placeholder="my_custom_tool")
        tool_description = st.text_area("Tool Description:", placeholder="Describe what your tool does", height=80)

    with col2:
        # Parameter definition
        st.markdown("**Parameters:**")
        param_name = st.text_input("Parameter Name:", placeholder="input_value")
        param_type = st.selectbox("Parameter Type:", ["string", "integer", "number", "boolean"])
        param_description = st.text_input("Parameter Description:", placeholder="Description of the parameter")

    # Tool code
    tool_code = st.text_area(
        "Tool Implementation (Python):",
        placeholder="""def my_tool(param1: str) -> str:
    # Your tool logic here
    return f"Processed: {param1}"
""",
        height=150,
    )

    # Test the custom tool
    if st.button("🧪 Test Custom Tool") and all([tool_name, tool_description, tool_code]):
        test_custom_tool(tool_name, tool_description, tool_code, param_name, param_type, param_description)

    # Predefined custom tool examples
    st.markdown("---")
    st.markdown("### 📋 Custom Tool Examples")

    custom_examples = {
        "Text Processor": {
            "name": "process_text",
            "description": "Process text in various ways",
            "code": """def process_text(text: str, operation: str = "upper") -> str:
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "reverse":
        return text[::-1]
    elif operation == "count":
        return f"Characters: {len(text)}, Words: {len(text.split())}"
    else:
        return text""",
            "query": "Process 'Hello World' by making it uppercase and then reverse it",
        },
        "Unit Converter": {
            "name": "convert_units",
            "description": "Convert between different units",
            "code": """def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    # Temperature conversions
    if from_unit == "celsius" and to_unit == "fahrenheit":
        return (value * 9/5) + 32
    elif from_unit == "fahrenheit" and to_unit == "celsius":
        return (value - 32) * 5/9
    # Length conversions
    elif from_unit == "meters" and to_unit == "feet":
        return value * 3.28084
    elif from_unit == "feet" and to_unit == "meters":
        return value / 3.28084
    else:
        return value""",
            "query": "Convert 25 degrees celsius to fahrenheit",
        },
    }

    for name, example in custom_examples.items():
        with st.expander(f"🔧 {name}"):
            st.code(example["code"], language="python")
            if st.button(f"Try: {example['query']}", key=f"custom_{name}"):
                # Create and test the example tool
                try:
                    exec(example["code"], globals())
                    func = globals()[example["name"]]
                    available_functions = {example["name"]: func}
                    tools = [func]
                    execute_tool_query(helper, model, example["query"], tools, available_functions, 0.1)
                except Exception as e:
                    st.error(f"Error executing custom tool: {e}")


def execute_tool_query(helper, model: str, query: str, tools: list, available_functions: dict, temperature: float):
    """Execute a query using tool calling."""

    st.markdown(f'### 🤖 Executing: "{query}"')

    with st.spinner("Processing with AI tools..."):
        try:
            messages = [{"role": "user", "content": query}]

            response, updated_messages = helper.chat_with_tools(
                model=model,
                messages=messages,
                tools=tools,
                available_functions=available_functions,
                options={"temperature": temperature},
            )

            # Display the response
            if response.message.content:
                st.success("✅ Task completed!")
                st.markdown("**AI Response:**")
                st.markdown(response.message.content)

            # Show tool calls if any
            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                with st.expander("🔧 Tool Calls Details"):
                    for i, tool_call in enumerate(response.message.tool_calls):
                        st.markdown(f"**Tool Call {i+1}:**")
                        st.write(f"Function: `{tool_call.function.name}`")
                        st.write(f"Arguments: `{tool_call.function.arguments}`")

                        # Execute and show result
                        if tool_call.function.name in available_functions:
                            try:
                                result = available_functions[tool_call.function.name](**tool_call.function.arguments)
                                st.write(f"Result: `{result}`")
                            except Exception as e:
                                st.write(f"Error: `{e}`")

            # Show conversation flow
            with st.expander("💬 Full Conversation"):
                for msg in updated_messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", str(msg))
                    st.write(f"**{role.title()}:** {content}")

        except Exception as e:
            st.error(f"Error executing tool query: {e}")


def test_custom_tool(name: str, description: str, code: str, param_name: str, param_type: str, param_desc: str):
    """Test a custom tool definition."""

    try:
        # Execute the tool code
        exec(code, globals())

        # Get the function
        if name in globals():
            func = globals()[name]
            st.success(f"✅ Tool '{name}' defined successfully!")

            # Show tool definition
            with st.expander("🔍 Tool Definition"):
                st.write(f"**Name:** {name}")
                st.write(f"**Description:** {description}")
                st.write(f"**Parameter:** {param_name} ({param_type}) - {param_desc}")
                st.code(code, language="python")

            # Test with sample input
            st.markdown("**Test your tool:**")
            test_input = st.text_input(f"Enter value for {param_name}:")

            if st.button("🧪 Test Tool") and test_input:
                try:
                    # Convert input based on type
                    test_value: Any = None
                    if param_type == "integer":
                        test_value = int(test_input)
                    elif param_type == "number":
                        test_value = float(test_input)
                    elif param_type == "boolean":
                        test_value = test_input.lower() in ["true", "1", "yes"]
                    else:
                        test_value = test_input

                    result = func(test_value)
                    st.success(f"Result: {result}")

                except Exception as e:
                    st.error(f"Tool execution error: {e}")
        else:
            st.error(f"Function '{name}' not found in code. Make sure the function name matches.")

    except Exception as e:
        st.error(f"Error defining tool: {e}")


if __name__ == "__main__":
    show()
