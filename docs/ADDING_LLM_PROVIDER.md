# Guide to Connecting a New LLM Provider

Step-by-step instructions for connecting another cloud LLM API or local LLM to this system.

---

## Core Principle: Provider Interface

All nodes (plan_generator, search_worker, writer, critic, etc.) depend only on the `LLMProvider` protocol. Create one class that implements this interface, register it in `main.py`, and you're done.

```python
# src/providers/base.py
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[dict],   # [{"role": "user", "content": "..."}]
        system: str = "",       # system prompt
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str: ...

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]: ...

    async def embed(self, text: str) -> list[float]: ...
```

> **Minimum implementation**: Only `complete()` is required.
> `stream()` is currently only used by the Chat graph, and `embed()` is only needed when using local file search. If not used, leave it as `raise NotImplementedError()`.

---

## Step 1: Write the Provider File

Create a new file in the `src/providers/` directory.

### Template

```python
# src/providers/my_provider.py
from __future__ import annotations
import os
from typing import AsyncIterator


class MyProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("MY_MODEL", "my-default-model")
        # SDK initialization
        # self._client = MySDK(api_key=api_key or os.environ["MY_API_KEY"])

    async def complete(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        # How system prompt is handled varies by SDK (see examples below)
        raise NotImplementedError

    async def stream(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        raise NotImplementedError

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Embeddings not supported")
```

---

## Step 2: Register in main.py

Add a branch to the `_create_llm_provider()` function in `src/main.py`.

```python
def _create_llm_provider(config: dict | None = None):
    ...
    elif provider == "my_provider":          # ← add
        from .providers.my_provider import MyProvider
        return MyProvider(
            model=config.get("model"),
            api_key=config.get("api_key"),
        )
    ...
```

Also add to `_get_llm_config_info()` in the same file.

```python
elif provider == "my_provider":
    config = {
        "provider": "my_provider",
        "model": os.getenv("MY_MODEL", "my-default-model"),
    }
```

---

## Step 3: Set Environment Variables

Add to the `.env` file.

```env
LLM_PROVIDER=my_provider
MY_MODEL=my-model-name
MY_API_KEY=sk-...
```

---

## Major Provider Implementation Examples

### OpenAI / Azure OpenAI

```python
# src/providers/openai_provider.py
from __future__ import annotations
import os
from typing import AsyncIterator
from openai import AsyncOpenAI


class OpenAIProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"]
        )

    async def complete(self, messages, system="", max_tokens=4096, temperature=0.3) -> str:
        # OpenAI inserts system as the first element of the messages array
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages, system="", max_tokens=4096, temperature=0.3):
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages

        async with self._client.chat.completions.stream(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as s:
            async for chunk in s:
                text = chunk.choices[0].delta.content or ""
                if text:
                    yield text

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding
```

**Azure OpenAI** uses `AsyncAzureOpenAI` instead of `AsyncOpenAI` and adds `api_version` and `azure_endpoint`.

```python
from openai import AsyncAzureOpenAI
self._client = AsyncAzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
)
```

Package installation: `pip install openai>=1.0`

---

### Google Gemini (Direct API)

```python
# src/providers/gemini_provider.py
from __future__ import annotations
import os
from typing import AsyncIterator
import google.generativeai as genai


class GeminiProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        genai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        self._client = genai.GenerativeModel(self.model)

    async def complete(self, messages, system="", max_tokens=4096, temperature=0.3) -> str:
        # Gemini passes system_instruction to the constructor or handles it separately
        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        # messages → Gemini ContentsType conversion
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages
        ]
        if system:
            # Workaround: prepend system before the first user turn
            contents[0]["parts"].insert(0, f"[System]\n{system}\n\n")

        response = await self._client.generate_content_async(
            contents, generation_config=config
        )
        return response.text

    async def stream(self, messages, system="", max_tokens=4096, temperature=0.3):
        config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens, temperature=temperature
        )
        contents = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in messages
        ]
        async for chunk in await self._client.generate_content_async(
            contents, generation_config=config, stream=True
        ):
            if chunk.text:
                yield chunk.text

    async def embed(self, text: str) -> list[float]:
        result = await genai.embed_content_async(
            model="models/text-embedding-004", content=text
        )
        return result["embedding"]
```

Package installation: `pip install google-generativeai>=0.8`

---

### Google Vertex AI (GCP Environment)

```python
# src/providers/vertexai_provider.py
import os
from typing import AsyncIterator
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


class VertexAIProvider:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("VERTEX_MODEL", "gemini-1.5-pro")
        vertexai.init(
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        self._client = GenerativeModel(self.model)

    async def complete(self, messages, system="", max_tokens=4096, temperature=0.3) -> str:
        config = GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        if system:
            prompt = f"[SYSTEM]\n{system}\n\n" + prompt
        response = await self._client.generate_content_async(prompt, generation_config=config)
        return response.text

    async def stream(self, messages, system="", max_tokens=4096, temperature=0.3):
        config = GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
        prompt = "\n\n".join(
            f"[{m['role'].upper()}]\n{m['content']}" for m in messages
        )
        async for chunk in await self._client.generate_content_async(
            prompt, generation_config=config, stream=True
        ):
            if chunk.text:
                yield chunk.text

    async def embed(self, text: str) -> list[float]:
        from vertexai.language_models import TextEmbeddingModel
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
```

Package installation: `pip install google-cloud-aiplatform>=1.60`

---

### Groq (Ultra-fast Inference API)

Uses the `openai` package as-is since it is an OpenAI-compatible API.

```python
# src/providers/groq_provider.py
import os
from typing import AsyncIterator
from openai import AsyncOpenAI


class GroqProvider:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )

    async def complete(self, messages, system="", max_tokens=4096, temperature=0.3) -> str:
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages, system="", max_tokens=4096, temperature=0.3):
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages
        stream = await self._client.chat.completions.create(
            model=self.model, messages=full_messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Groq does not support embeddings — use Ollama or OpenAI embed")
```

Package installation: `pip install openai>=1.0`

---

### LM Studio (Local OpenAI-Compatible Server)

LM Studio has a built-in OpenAI-compatible server. Just change the `base_url`.

```python
# src/providers/lmstudio_provider.py
import os
from typing import AsyncIterator
from openai import AsyncOpenAI


class LMStudioProvider:
    """
    Connects to LM Studio's local OpenAI-compatible server.
    Requires enabling LM Studio → Developer → Start Server.
    """
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("LMSTUDIO_MODEL", "local-model")
        _host = host or os.getenv("LMSTUDIO_HOST", "http://localhost:1234")
        self._client = AsyncOpenAI(
            api_key="lm-studio",          # value doesn't matter, but cannot be empty
            base_url=f"{_host}/v1",
        )

    async def complete(self, messages, system="", max_tokens=4096, temperature=0.3) -> str:
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, messages, system="", max_tokens=4096, temperature=0.3):
        full_messages = messages
        if system:
            full_messages = [{"role": "system", "content": system}] + messages
        stream = await self._client.chat.completions.create(
            model=self.model, messages=full_messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
        )
        async for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                yield text

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self.model, input=text
        )
        return response.data[0].embedding
```

---

### vLLM (OpenAI-Compatible Server)

vLLM also provides an OpenAI-compatible server. The structure is identical to LM Studio.

```bash
# vLLM server launch example
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-72B-Instruct \
  --host 0.0.0.0 --port 8001
```

```python
# src/providers/vllm_provider.py
import os
from openai import AsyncOpenAI


class VLLMProvider:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        _host = host or os.getenv("VLLM_HOST", "http://localhost:8001")
        self._client = AsyncOpenAI(api_key="vllm", base_url=f"{_host}/v1")

    # complete(), stream(), embed() — same implementation as LMStudioProvider
```

---

## Comparison of System Prompt Handling Methods

The way system prompts are passed differs across LLM APIs.

| Provider | System prompt handling |
|-----------|----------------|
| **Anthropic Claude** | Separate `system=` parameter field in API |
| **OpenAI / Groq / vLLM** | First element `{"role": "system", ...}` in `messages` array |
| **Ollama** | First element `{"role": "system", ...}` in `messages` array |
| **Gemini** | `system_instruction` parameter or prepended to first user turn |
| **AWS Bedrock** | Separate `system=` parameter field in API (same as Claude) |

This system has each node pass `system=` as a separate argument, so the provider implementation internally converts it to the appropriate SDK format.

---

## When embed() Is Needed

`embed()` is only called when using local file search (`/files/index`). If not used, it is fine to leave it as `raise NotImplementedError()`.

Example embedding model combinations:

| LLM | Embeddings |
|-----|--------|
| OpenAI GPT-4o | `text-embedding-3-small` (same API) |
| Groq (inference only) | Recommended to use Ollama `nomic-embed-text` in parallel |
| Gemini | `text-embedding-004` (same API) |
| Local (vLLM/LM Studio) | Ollama `nomic-embed-text` or fastembed direct use |

If you want to use Ollama embeddings separately, you can borrow and combine the `embed()` method from `OllamaProvider`.

---

## Adding a New Provider to the UI Settings Modal (Optional)

To support runtime switching from the UI via `POST /settings`, also modify the frontend.

**`frontend/src/components/SettingsModal.tsx`** — add an item to the radio buttons:

```tsx
const PROVIDERS = [
  { value: "bedrock", label: "AWS Bedrock" },
  { value: "claude",  label: "Anthropic Claude" },
  { value: "ollama",  label: "Ollama (Local)" },
  { value: "openai",  label: "OpenAI" },       // ← add
  { value: "groq",    label: "Groq" },          // ← add
  { value: "lmstudio", label: "LM Studio" },   // ← add
];
```

---

## Checklist

Items to verify when adding a new provider:

- [ ] Create file in `src/providers/`
- [ ] Implement `complete()` (required)
- [ ] Implement `stream()` (if using Chat feature)
- [ ] Implement `embed()` or leave as `NotImplementedError` (not needed if not using local file search)
- [ ] Add branch to `_create_llm_provider()` in `src/main.py`
- [ ] Add branch to `_get_llm_config_info()` in `src/main.py`
- [ ] Add new environment variables to `.env.example`
- [ ] Add required packages to `requirements.txt`

---

## Debugging Tips

**When JSON parsing fails frequently**: The model is not following instructions well. The DSAP Guard Functions in `src/utils/llm_json.py` automatically retry, but the model's own instruction-following performance matters. Choose a model strong at JSON output (Qwen3, GPT-4o series).

**When Korean quality is poor**: All prompts are written in Korean. Choose a model with sufficient Korean training data (Qwen3, EXAONE, HyperCLOVA X, etc.).

**Ollama timeouts**: Even in `detailed` report mode the system is designed to run sequentially, but if the model is very slow, you can increase the `timeout` parameter when initializing `AsyncClient` in `src/providers/ollama.py`.

```python
self._client = AsyncClient(host=_host, timeout=600)  # 10 minutes
```
