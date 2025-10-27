"""
fetch_models.py
Fetch current Ollama model list from https://ollama.com/search
and generate a Python module at <script_parent>/lib/models.py
containing `popular_models`.
"""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
#  Heuristic keyword groups
# ---------------------------------------------------------------------------

EMBED_KWS = {"embed", "minilm", "e5", "gte", "bge", "nomic", "text-embedding"}
VISION_KWS = {"vision", "llava", "bakllava", "moondream", "cogvlm", "qwen-vl"}
CODE_KWS = {"code", "codellama", "codegemma", "starcoder", "wizardcoder", "replit"}
AUDIO_KWS = {"whisper", "wav2vec", "audio"}
TOOLING_KWS = {"function", "tool", "agent", "assistant", "qwen2.5", "llama3.2"}
MULTIMODAL_KWS = {"vl", "mm", "multimodal"}

# ---------------------------------------------------------------------------
#  Fetch model list from Ollama
# ---------------------------------------------------------------------------


def fetch_ollama_models(timeout: int = 15) -> set[str]:
    """
    Scrape model names from https://ollama.com/search
    Returns a set of lowercase model names like {"llama3.2:3b", "gemma3:9b"}.
    """
    url = "https://ollama.com/search"
    headers = {"User-Agent": "Mozilla/5.0 (Python fetch_models.py)"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        models = set()
        for a in soup.find_all("a", href=True):
            m = re.search(r"/library/([a-z0-9][\w\-\.:]+)", a["href"])
            if m:
                models.add(m.group(1).lower())

        if not models:
            raise ValueError("No models found in page")

        return models

    except Exception as e:
        print(f"[WARN] Could not fetch live model list: {e}")
        # fallback list
        return {
            "llama3.2",
            "llama3.2:3b",
            "llama3.1:8b",
            "mistral:7b",
            "mixtral:8x7b",
            "gemma3:2b",
            "llava:7b",
            "nomic-embed-text",
            "codellama:7b-code",
        }


# ---------------------------------------------------------------------------
#  Categorization
# ---------------------------------------------------------------------------


def categorize_model(name: str) -> str:
    n = name.lower()
    if any(k in n for k in EMBED_KWS):
        return "Embedding Models"
    if any(k in n for k in VISION_KWS):
        return "Vision Models"
    if any(k in n for k in CODE_KWS):
        return "Code Models"
    if any(k in n for k in AUDIO_KWS):
        return "Audio / Speech"
    if any(k in n for k in TOOLING_KWS):
        return "Tooling / Agents"
    if any(k in n for k in MULTIMODAL_KWS):
        return "Multimodal Models"
    return "Text Models"


# ---------------------------------------------------------------------------
#  Build grouped dictionary
# ---------------------------------------------------------------------------


def build_popular_models() -> dict[str, list[str]]:
    models = fetch_ollama_models()
    grouped: dict[str, list[str]] = {}

    # Add the "All" group containing all models
    grouped["All"] = sorted(set(models))

    for m in sorted(models):
        cat = categorize_model(m)
        grouped.setdefault(cat, []).append(m)

    # Sort alphabetically inside groups
    for k in grouped:
        grouped[k] = sorted(set(grouped[k]))

    return grouped


# ---------------------------------------------------------------------------
#  Write models.py to parent/lib/
# ---------------------------------------------------------------------------


def write_python_module(data: dict[str, list[str]]):
    """
    Write the grouped model data as a valid Python dictionary to
    <parent_of_this_script>/lib/models.py
    """
    script_dir = Path(__file__).resolve().parent.parent
    target_path = script_dir / "lib" / "helper_models" / "__init__.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as f:
        f.write("models = ")
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")

    total_models = sum(len(v) for k, v in data.items() if k != "All")
    print(f"[OK] Wrote {target_path} with {total_models} categorized models and {len(data['All'])} total in 'All'.")


# ---------------------------------------------------------------------------
#  Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grouped = build_popular_models()
    write_python_module(grouped)
