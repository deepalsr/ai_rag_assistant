"""
LLM GATEWAY
------------
Job: sit between our app and the LLM provider(s), adding rate limiting,
fallback routing, and cost/usage tracking. Nothing here changes WHAT
question gets answered - only HOW reliably and safely we call the LLM.
"""

import time


class RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float):
        self.max_calls = max_calls
        self.per_seconds = per_seconds
        self.call_timestamps = []

    def allow(self) -> bool:
        """Check if a new call is allowed right now under the rate limit."""
        now = time.time()
        self.call_timestamps = [t for t in self.call_timestamps if now - t < self.per_seconds]

        if len(self.call_timestamps) < self.max_calls:
            self.call_timestamps.append(now)
            return True
        return False


class LLMGateway:
    def __init__(self, client, models: list[str], rate_limiter: RateLimiter):
        self.client = client
        self.models = models
        self.rate_limiter = rate_limiter
        self.usage_log = []

    PRICING = {
        "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
        "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
    }

    def call(self, prompt: str, temperature: float = 0.1) -> dict:
        if not self.rate_limiter.allow():
            raise RuntimeError("Rate limit exceeded - too many requests in this window")

        last_error = None
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                self._log_usage(model, response.usage)
                return {
                    "text": response.choices[0].message.content,
                    "model_used": model,
                    "usage": response.usage,
                }
            except Exception as e:
                print(f"⚠️  [gateway] {model} failed: {e}. Trying next model...")
                last_error = e
                continue

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def _log_usage(self, model: str, usage) -> None:
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        input_cost = (usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1_000_000) * pricing["output"]

        self.usage_log.append({
            "model": model,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "cost_usd": input_cost + output_cost,
        })

    def total_cost(self) -> float:
        return sum(entry["cost_usd"] for entry in self.usage_log)

    def summary(self) -> None:
        print(f"\n--- Gateway usage summary ---")
        print(f"Total calls: {len(self.usage_log)}")
        print(f"Total cost: ${self.total_cost():.6f}")
        for model in set(e["model"] for e in self.usage_log):
            count = sum(1 for e in self.usage_log if e["model"] == model)
            print(f"  {model}: {count} calls")