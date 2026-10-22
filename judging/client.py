import asyncio
import json
import time
from typing import Optional

from openai import APIError, APITimeoutError, AsyncOpenAI

from .models import SAMPLING, ModelConfig

MAX_RETRIES = 5
RETRY_DELAY = 5


class LLMClient:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.client = AsyncOpenAI(
            base_url=model_config.endpoint,
            api_key=model_config.api_key,
            timeout=240.0,
        )

    async def generate(
        self, prompt: str, seed: int, schema: Optional[dict] = None
    ) -> tuple[str, dict]:
        start_time = time.time()

        kwargs = {
            "model": self.config.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": SAMPLING.max_tokens,
            "seed": seed,
        }

        if schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "schema": schema,
                    "strict": True,
                },
            }

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.chat.completions.create(**kwargs)

                text = response.choices[0].message.content.strip()
                elapsed = time.time() - start_time
                metadata = {
                    "model": self.config.name,
                    "seed": seed,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    "elapsed_time": elapsed,
                }
                return text, metadata

            except (APITimeoutError, APIError) as e:
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"[{self.config.short_name}] Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {RETRY_DELAY}s..."
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(
                        f"[{self.config.short_name}] All {MAX_RETRIES} attempts failed: {e}"
                    )
                    raise


def parse_json_response(response: str) -> Optional[dict]:
    if not response:
        return None

    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        candidate = response[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None

    return None
