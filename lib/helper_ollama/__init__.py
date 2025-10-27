import os
import base64

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Generator
from pydantic import BaseModel

import threading

from ollama import (
    Client,
    AsyncClient,
    ChatResponse,
    ListResponse,
    ShowResponse,
    ProcessResponse,
    list,
    chat,
    generate,
    embed,
    show,
    pull,
    ps,
    create,
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

DEFAULT_MODEL = "phi4-mini"


class OllamaHelper:
    """Singleton helper class for Ollama operations with both sync and async methods."""

    _instance_lock = threading.Lock()
    _instance = None

    def __new__(cls, base_url: Optional[str] = None):
        # Double-checked locking to ensure singleton
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # store the base_url used for the first initialization
                    cls._instance._initial_base_url = base_url

        return cls._instance

    def __init__(self, base_url: Optional[str] = None):
        # Avoid re-initialization on subsequent constructions
        if getattr(self, "_initialized", False):
            return

        # Respect the base_url provided on first construction only
        _base_url = getattr(self, "_initial_base_url", base_url)

        self.client = Client(host=_base_url) if _base_url else Client()
        self.client_async = AsyncClient(host=_base_url) if _base_url else AsyncClient()

        self._initialized = True

    @classmethod
    def get_instance(cls, base_url: Optional[str] = None) -> "OllamaHelper":
        """
        Get the singleton instance. If the singleton doesn't exist, it will be created.
        The base_url argument is only applied on the first creation call.
        """
        return cls(base_url)

    # ============================================================================
    # CHAT OPERATIONS
    # ============================================================================

    def simple_chat(self, model: str, message: str, **kwargs) -> str:
        messages = [{"role": "user", "content": message}]
        response = self.client.chat(model, messages=messages, **kwargs)
        return response["message"]["content"]

    async def async_simple_chat(self, model: str, message: str, **kwargs) -> str:
        messages = [{"role": "user", "content": message}]
        response = await self.client_async.chat(model, messages=messages, **kwargs)
        return response["message"]["content"]

    def chat_with_messages(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> ChatResponse:
        return chat(model, messages=messages, **kwargs)

    async def async_chat_with_messages(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> ChatResponse:
        return await self.client_async.chat(model, messages=messages, **kwargs)

    def chat_stream(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        for part in chat(model, messages=messages, stream=True, **kwargs):
            yield part

    def chat_with_history(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> tuple[str, List[Dict[str, str]]]:
        response = chat(model, messages=messages, **kwargs)
        messages.append(response["message"])
        return response["message"]["content"], messages

    # ============================================================================
    # MULTIMODAL OPERATIONS
    # ============================================================================

    def multimodal_chat(
        self, model: str, message: str, image_path: str, **kwargs
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": message,
                "images": [image_path],
            }
        ]
        response = chat(model, messages=messages, **kwargs)
        return response["message"]["content"]

    def multimodal_chat_base64(
        self, model: str, message: str, image_base64: str, **kwargs
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": message,
                "images": [image_base64],
            }
        ]
        response = chat(model, messages=messages, **kwargs)
        return response["message"]["content"]

    def encode_image_to_base64(self, image_path: str) -> str:
        return base64.b64encode(Path(image_path).read_bytes()).decode()

    # ============================================================================
    # GENERATION OPERATIONS
    # ============================================================================

    def simple_generate(self, model: str, prompt: str, **kwargs) -> str:
        response = generate(model, prompt, **kwargs)
        return response["response"]

    async def async_simple_generate(self, model: str, prompt: str, **kwargs) -> str:
        response = await self.client_async.generate(model, prompt, **kwargs)
        return response["response"]

    def generate_stream(
        self, model: str, prompt: str, **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        for part in generate(model, prompt, stream=True, **kwargs):
            yield part

    # ============================================================================
    # EMBEDDINGS
    # ============================================================================

    def get_embeddings(self, model: str, input_text: str, **kwargs) -> List[float]:
        response = embed(model=model, input=input_text, **kwargs)
        return response["embeddings"]

    def get_embeddings_batch(
        self, model: str, input_texts: List[str], **kwargs
    ) -> List[List[float]]:
        response = embed(model=model, input=input_texts, **kwargs)
        return response["embeddings"]

    # ============================================================================
    # MODEL MANAGEMENT
    # ============================================================================
    def models_list(self, with_details: bool = False) -> ListResponse:
        models: ListResponse = list().models

        if with_details:
            return models
        else:
            return [m.model for m in models]
        
    def get_model_info(self, model: str) -> ShowResponse:
        return show(model)

    def pull_model(
        self, model: str, stream: bool = False
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        return pull(model, stream=stream)

    def pull_model_with_progress(self, model: str) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            print("tqdm not installed, showing basic progress")
            for progress in pull(model, stream=True):
                print(progress.get("status", ""))
            return

        current_digest, bars = "", {}
        for progress in pull(model, stream=True):
            digest = progress.get("digest", "")
            if digest != current_digest and current_digest in bars:
                bars[current_digest].close()

            if not digest:
                print(progress.get("status"))
                continue

            if digest not in bars and (total := progress.get("total")):
                bars[digest] = tqdm(
                    total=total,
                    desc=f"pulling {digest[7:19]}",
                    unit="B",
                    unit_scale=True,
                )

            if completed := progress.get("completed"):
                bars[digest].update(completed - bars[digest].n)

            current_digest = digest

    def list_running_models(self) -> ProcessResponse:
        response = ps()
        return response.models


    def create_model(
        self,
        name: str,
        from_model: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        params = {"model": name, "from_": from_model, "stream": stream, **kwargs}
        if system_prompt:
            params["system"] = system_prompt
        return create(**params)

    # ============================================================================
    # TOOL CALLING
    # ============================================================================

    def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: List[Union[Callable, Dict]],
        available_functions: Dict[str, Callable],
        **kwargs,
    ) -> tuple[ChatResponse, List[Dict[str, str]]]:
        response: ChatResponse = chat(model, messages=messages, tools=tools, **kwargs)

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    output = function_to_call(**tool.function.arguments)
                    messages.append(response.message)
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(output),
                            "tool_name": tool.function.name,
                        }
                    )

            final_response = chat(model, messages=messages)
            return final_response, messages

        return response, messages

    async def async_chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: List[Union[Callable, Dict]],
        available_functions: Dict[str, Callable],
        **kwargs,
    ) -> tuple[ChatResponse, List[Dict[str, str]]]:
        response: ChatResponse = await self.client_async.chat(
            model, messages=messages, tools=tools, **kwargs
        )

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    output = function_to_call(**tool.function.arguments)
                    messages.append(response.message)
                    messages.append(
                        {
                            "role": "tool",
                            "content": str(output),
                            "tool_name": tool.function.name,
                        }
                    )

            final_response = await self.client_async.chat(model, messages=messages)
            return final_response, messages

        return response, messages

    def create_tool_definition(
        self, name: str, description: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    # ============================================================================
    # STRUCTURED OUTPUTS
    # ============================================================================

    def chat_with_structured_output(
        self,
        model: str,
        messages: List[Dict[str, str]],
        schema: Union[BaseModel, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            format_schema = schema.model_json_schema()
            response = chat(model, messages=messages, format=format_schema, **kwargs)
            return schema.model_validate_json(response.message.content)
        else:
            response = chat(model, messages=messages, format=schema, **kwargs)
            return json.loads(response.message.content)

    def generate_with_structured_output(
        self,
        model: str,
        prompt: str,
        schema: Union[BaseModel, Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            format_schema = schema.model_json_schema()
            response = generate(model, prompt, format=format_schema, **kwargs)
            return schema.model_validate_json(response["response"])
        else:
            response = generate(model, prompt, format=schema, **kwargs)
            return json.loads(response["response"])

    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================

    def print_model_list(self) -> None:
        response = self.models_list()
        # If with_details was used, response may be a ListResponse object
        if hasattr(response, "models"):
            for model in response.models:
                print(f"Name: {model.model}")
                print(f"  Size (MB): {(float(model.size) / 1024 / 1024):.2f}")
                if model.details:
                    print(f"  Format: {model.details.format}")
                    print(f"  Family: {model.details.family}")
                    print(f"  Parameter Size: {model.details.parameter_size}")
                    print(f"  Quantization Level: {model.details.quantization_level}")
                print()
        else:
            for name in response:
                print(name)

    def print_model_info(self, model: str) -> None:
        response = self.get_model_info(model)
        print("Model Information:")
        print(f"Modified at:   {response.modified_at}")
        print(f"Template:      {response.template}")
        print(f"Modelfile:     {response.modelfile}")
        print(f"License:       {response.license}")
        print(f"Details:       {response.details}")
        print(f"Model Info:    {response.modelinfo}")
        print(f"Parameters:    {response.parameters}")
        print(f"Capabilities:  {response.capabilities}")

    def print_running_models(self) -> None:
        response = self.list_running_models()
        for model in response.models:
            print(f"Model: {model.model}")
            print(f"  Digest: {model.digest}")
            print(f"  Expires at: {model.expires_at}")
            print(f"  Size: {model.size}")
            print(f"  Size vram: {model.size_vram}")
            print(f"  Details: {model.details}")
            print(f"  Context length: {model.context_length}")
            print()

    def stream_chat_to_console(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        content = ""
        for part in self.chat_stream(model, messages, **kwargs):
            part_content = part["message"]["content"]
            print(part_content, end="", flush=True)
            content += part_content
        print()
        return content

    def ensure_model_loaded(self, model: str) -> bool:
        try:
            self.simple_chat(model, "Hello")
            return True
        except Exception as e:
            print(f"Failed to load model {model}: {e}")
            return False

helper = OllamaHelper()