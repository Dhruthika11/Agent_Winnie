"""
Multi-Provider Fallback Router and Token Pooling for LiteLLM.

Provides:
- Token pooling across multiple model deployments and API keys
- Dynamic usage-based routing (routes to least utilized deployment within TPM budgets)
- Automatic cross-provider and cross-model failover upon 429 rate limits or provider outages
- Cooldown tracking for failing/exhausted endpoints
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from litellm import Router

from src.config import settings

logger = logging.getLogger("hermes.agent.router")

LOGICAL_AGENT_MODEL = "winnie-agent-model"

# Default TPM (Tokens Per Minute) and RPM limits for common free/standard tiers
MODEL_LIMIT_DEFAULTS = {
    # Groq models (Standard tier allocation)
    "groq/qwen/qwen3.8-27b": {"tpm": 40000, "rpm": 60},
    "groq/openai/gpt-oss-120b": {"tpm": 40000, "rpm": 60},
    "groq/openai/gpt-oss-20b": {"tpm": 40000, "rpm": 60},
    "groq/llama-3.3-70b-versatile": {"tpm": 30000, "rpm": 30},
    "groq/llama-3.1-8b-instant": {"tpm": 40000, "rpm": 60},
    "groq/llama3-70b-8192": {"tpm": 30000, "rpm": 30},
    "groq/mixtral-8x7b-32768": {"tpm": 30000, "rpm": 30},
    # Google Gemini Multimodal models
    "gemini/gemini-1.5-flash": {"tpm": 1000000, "rpm": 15},
    "gemini/gemini-1.5-pro": {"tpm": 2000000, "rpm": 2},
    "gemini/gemini-2.0-flash": {"tpm": 1000000, "rpm": 15},
    "gemini/gemini-2.0-flash-exp": {"tpm": 1000000, "rpm": 10},
}

# ---------------------------------------------------------------------------
# Fallback Performance Hierarchy Index (excluding vision models)
# 1 = Highest performance & reasoning capability; higher index = smaller/faster fallbacks.
# ---------------------------------------------------------------------------
MODEL_PERFORMANCE_INDEX: dict[str, dict[str, Any]] = {
    "groq/openai/gpt-oss-120b": {
        "index": 1,
        "tier": "Tier 1: Ultra-Large 120B (High Complexity, Architecture & Broad Knowledge)",
        "parameter_class": "120B",
    },
    "groq/qwen/qwen3.8-27b": {
        "index": 2,
        "tier": "Tier 2: Specialized 27B (Advanced Coding, Syntax & Tool Accuracy)",
        "parameter_class": "27B",
    },
    "groq/openai/gpt-oss-20b": {
        "index": 3,
        "tier": "Tier 3: Balanced 20B (Ultra-Fast Synthesis, Chat & Low Latency)",
        "parameter_class": "20B",
    },
    "groq/llama-3.3-70b-versatile": {
        "index": 4,
        "tier": "Tier 4: Llama 70B (Fallback Flagship)",
        "parameter_class": "70B",
    },
    "groq/llama-3.1-8b-instant": {
        "index": 5,
        "tier": "Tier 5: Llama 8B (Fallback Fast)",
        "parameter_class": "8B",
    },
    "groq/llama3-70b-8192": {
        "index": 6,
        "tier": "Tier 6: Legacy 70B",
        "parameter_class": "70B",
    },
    "groq/mixtral-8x7b-32768": {
        "index": 7,
        "tier": "Tier 7: Mixture-of-Experts 8x7B",
        "parameter_class": "8x7B",
    },
}

# ---------------------------------------------------------------------------
# Task-Based Model Hierarchies
# Different tasks prioritize different model strengths:
# - Coding: Qwen 27B -> GPT-OSS 120B -> GPT-OSS 20B
# - Reasoning: GPT-OSS 120B -> Qwen 27B -> GPT-OSS 20B
# - Knowledge / Summarization: GPT-OSS 120B -> GPT-OSS 20B -> Qwen 27B
# - Quick Chat / Low Latency: GPT-OSS 20B -> Qwen 27B -> GPT-OSS 120B
# - Multimodal: Gemini 1.5 Flash (Vision & Audio OCR/Inspection)
# ---------------------------------------------------------------------------
TASK_PROFILES: dict[str, dict[str, Any]] = {
    "coding": {
        "label": "Coding, Tool Execution & Structured Data",
        "description": "Specialized for Python, scripts, MCP tool JSON schemas, debugging, and terminal operations",
        "primary_model": "groq/qwen/qwen3.8-27b",
        "hierarchy": [
            "groq/qwen/qwen3.8-27b",
            "groq/openai/gpt-oss-120b",
            "groq/openai/gpt-oss-20b",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
        ],
    },
    "reasoning": {
        "label": "Complex Reasoning, Architecture & Multi-Step Logic",
        "description": "Specialized for algorithmic problem solving, trade-off analysis, system design, and planning",
        "primary_model": "groq/openai/gpt-oss-120b",
        "hierarchy": [
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3.8-27b",
            "groq/openai/gpt-oss-20b",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
        ],
    },
    "knowledge_synthesis": {
        "label": "Knowledge Retrieval, Research & Summarization",
        "description": "Specialized for broad domain knowledge, long-form documents, historical synthesis, and report generation",
        "primary_model": "groq/openai/gpt-oss-120b",
        "hierarchy": [
            "groq/openai/gpt-oss-120b",
            "groq/qwen/qwen3.8-27b",
            "groq/openai/gpt-oss-20b",
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
        ],
    },
    "quick_chat": {
        "label": "Fast Conversational & Low-Latency Query",
        "description": "Specialized for greetings, quick factoids, conversational chat, and preserving TPM quota",
        "primary_model": "groq/openai/gpt-oss-20b",
        "hierarchy": [
            "groq/openai/gpt-oss-20b",
            "groq/qwen/qwen3.8-27b",
            "groq/openai/gpt-oss-120b",
            "groq/llama-3.1-8b-instant",
            "groq/llama-3.3-70b-versatile",
        ],
    },
    "multimodal": {
        "label": "Multimodal Vision & Audio",
        "description": "Specialized for image recognition, OCR, screenshot analysis, and diagram interpretation",
        "primary_model": "gemini/gemini-1.5-flash",
        "hierarchy": [
            "gemini/gemini-1.5-flash",
            "gemini/gemini-2.0-flash",
            "gemini/gemini-1.5-pro",
        ],
    },
}


def classify_task_intent(prompt: str, tools: list[dict[str, Any]] | None = None, has_images: bool = False) -> str:
    """Classify the user intent into one of the specialized task profiles in sub-millisecond time."""
    if has_images:
        return "multimodal"

    lower = (prompt or "").lower().strip()

    # Fast conversational greeting / short query check
    greetings = {"hi", "hello", "hey", "good morning", "good evening", "howdy", "sup", "thanks", "thank you", "bye", "who are you"}
    if lower in greetings or (len(lower) < 25 and any(lower.startswith(g) for g in ["hi", "hello", "hey", "yo"])):
        return "quick_chat"

    # Fast memory / user preference instruction check
    memory_patterns = [
        r"^remember\b",
        r"\bkeep\s+in\s+mind\b",
        r"\bplease\s+note\b",
        r"^note\s+that\b",
        r"\bmy\s+team\b",
        r"\bmy\s+preference\b",
        r"\bi\s+prefer\b",
        r"\bi\s+strictly\s+use\b",
        r"\bdo\s+not\s+forget\b",
        r"\bdon't\s+forget\b",
    ]
    if any(re.search(p, lower) for p in memory_patterns):
        return "quick_chat"

    # 1. Complex reasoning, architecture, trade-offs, and multi-step planning
    reasoning_indicators = [
        "trade-off", "tradeoff", "pros and cons", "architecture", "system design",
        "compare", "comparison", "evaluate", "evaluation", "analyze", "analysis",
        "why", "plan", "strategy", "step by step", "solve", "algorithm", "logic", "proof"
    ]
    if any(k in lower for k in reasoning_indicators):
        return "reasoning"

    # 2. Knowledge synthesis, broad research, and summarization
    synthesis_indicators = [
        "summarize", "summary", "research", "overview", "history", "essay",
        "article", "report", "documentation", "explain what is", "what are all", "bullet points"
    ]
    if any(k in lower for k in synthesis_indicators):
        return "knowledge_synthesis"

    # 3. Coding & MCP Tool execution indicators (use word boundary regex to avoid false positives like 'sql' in 'postgresql')
    code_word_patterns = [
        r"\bpython\b", r"\bjavascript\b", r"\btypescript\b", r"\bhtml\b", r"\bcss\b",
        r"\bsql\b", r"\bbash\b", r"\bshell\b", r"\bcode\b", r"\bdef\s+", r"\bfunction\b",
        r"\bclass\s+", r"\bimport\s+", r"\bregex\b", r"\bapi\b", r"\bjson\b", r"\bschema\b",
        r"\bdebug\b", r"\bbug\b", r"\brefactor\b", r"\bunit\s+test\b", r"\bgit\b", r"\bcommit\b",
        r"\bterminal\b", r"\bcompile\b", r"\bscript\b", r"\bcurl\b", r"\bendpoint\b", r"\brepo\b",
        r"\brepository\b", r"\bdocker\b", r"\bfastapi\b", r"\blist_directory\b", r"\bread_file\b",
        r"^(?:sudo\s+)?(?:rm|del|delete|unlink|rmdir|mkdir|cat|ls|dir|chmod|chown|mv|cp|touch|grep|find|kill|ps|tar|zip|unzip|curl|wget|echo|sh|bash|zsh|pwsh|powershell|cmd)\b",
        r"\b(?:rm|del|delete|unlink|rmdir|mkdir|chmod|chown)\s+(?:-[a-zA-Z]+\s+)*[.\/~]",
        r"\b(?:terminal|command|cli|run\s+command|execute|delete\s+file)\b",
        r"\.(?:log|txt|py|json|yaml|yml|sh|md|tmp)\b",
    ]
    if any(re.search(p, lower) for p in code_word_patterns):
        return "coding"

    if len(lower) < 45:
        return "quick_chat"
    return "reasoning"


_PROXY_HEALTH_CACHE: dict[str, tuple[bool, float]] = {}


def _is_proxy_responsive(url: str, timeout_sec: float = 0.5, cache_ttl: float = 5.0) -> bool:
    """
    Robust health probe to test if LiteLLM proxy is responsive before dispatching.
    Caches status for cache_ttl seconds. Uses strict connection and read timeouts
    to detect zombie sockets and half-open TCP states without hanging.
    """
    if not url:
        return False
    now = time.time()
    cached = _PROXY_HEALTH_CACHE.get(url)
    if cached and (now - cached[1]) < cache_ttl:
        return cached[0]

    is_healthy = False
    clean_url = url.rstrip("/")
    headers = {"User-Agent": "Hermes-HealthCheck/1.0"}
    if settings.litellm_master_key:
        headers["Authorization"] = f"Bearer {settings.litellm_master_key}"

    for probe_endpoint in ("/health/readiness", "/health", "/"):
        try:
            import urllib.request
            health_url = f"{clean_url}{probe_endpoint}"
            req = urllib.request.Request(health_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                if resp.status < 500:
                    is_healthy = True
                    break
        except Exception:
            continue

    _PROXY_HEALTH_CACHE[url] = (is_healthy, now)
    return is_healthy


def _build_deployments() -> list[dict[str, Any]]:
    """Build the list of model deployments for the token pool."""
    deployments: list[dict[str, Any]] = []

    # Check if LiteLLM proxy is actually alive and responsive
    proxy_alive = bool(settings.litellm_base_url and _is_proxy_responsive(settings.litellm_base_url))
    if settings.litellm_base_url and not proxy_alive:
        logger.warning(
            f"LiteLLM proxy configured at {settings.litellm_base_url} is not responding within timeout. "
            "Bypassing proxy to route directly to upstream model providers."
        )

    # 1. Groq Deployments (Primary and Fallbacks across keys)
    groq_keys = []
    if settings.groq_api_keys:
        groq_keys.extend([k.strip() for k in settings.groq_api_keys.split(",") if k.strip()])
    if settings.litellm_master_key and settings.litellm_master_key.startswith("gsk_"):
        if settings.litellm_master_key not in groq_keys:
            groq_keys.insert(0, settings.litellm_master_key)
    
    # Check OS env if not in settings
    env_groq_key = os.environ.get("GROQ_API_KEY")
    if env_groq_key and env_groq_key not in groq_keys:
        groq_keys.append(env_groq_key)

    candidate_models: list[str] = []
    if settings.model_name and settings.model_name not in candidate_models:
        candidate_models.append(settings.model_name)
    if settings.fallback_models:
        for m in settings.fallback_models.split(","):
            m = m.strip()
            if m and m not in candidate_models:
                candidate_models.append(m)

    # Sort deployed text models strictly by their performance index (Tier 1 -> Tier N)
    def _model_performance_sort_key(m: str) -> int:
        return MODEL_PERFORMANCE_INDEX.get(m, {}).get("index", 999)

    groq_models = sorted(candidate_models, key=_model_performance_sort_key)

    for key_idx, key in enumerate(groq_keys):
        masked_key = f"{key[:7]}...{key[-4:]}" if len(key) > 12 else "key"
        for rank_idx, model in enumerate(groq_models, start=1):
            limits = MODEL_LIMIT_DEFAULTS.get(model, {"tpm": 7000, "rpm": 30})
            perf_info = MODEL_PERFORMANCE_INDEX.get(model, {
                "index": rank_idx,
                "tier": f"Tier {rank_idx}: Standard Agent Model",
                "parameter_class": "General",
            })
            # A. LiteLLM Proxy Deployment (only if responsive)
            if proxy_alive:
                proxy_model = f"openai/{model}" if not model.startswith("openai/") else model
                proxy_key = settings.litellm_master_key or key
                proxy_params: dict[str, Any] = {
                    "model": proxy_model,
                    "api_key": proxy_key,
                    "api_base": settings.litellm_base_url,
                    "tpm": limits["tpm"],
                    "rpm": limits["rpm"],
                }
                deployments.append({
                    "model_name": LOGICAL_AGENT_MODEL,
                    "litellm_params": proxy_params,
                    "model_info": {
                        "provider": "litellm-proxy",
                        "real_model": model,
                        "key_identifier": f"proxy-key-{key_idx+1}-{masked_key}",
                        "performance_index": perf_info["index"],
                        "performance_tier": perf_info["tier"],
                        "parameter_class": perf_info.get("parameter_class", "General"),
                        "multimodal": False,
                    }
                })

            # B. Direct Provider Deployment (used directly or as auto-fallback if proxy is unreachable)
            direct_params: dict[str, Any] = {
                "model": model,
                "api_key": key,
                "tpm": limits["tpm"],
                "rpm": limits["rpm"],
            }
            deployments.append({
                "model_name": LOGICAL_AGENT_MODEL,
                "litellm_params": direct_params,
                "model_info": {
                    "provider": "groq",
                    "real_model": model,
                    "key_identifier": f"groq-key-{key_idx+1}-{masked_key}",
                    "performance_index": perf_info["index"],
                    "performance_tier": perf_info["tier"],
                    "parameter_class": perf_info.get("parameter_class", "General"),
                    "multimodal": False,
                }
            })
            deployments.append({
                "model_name": model,
                "litellm_params": direct_params,
                "model_info": {
                    "provider": "groq",
                    "real_model": model,
                    "key_identifier": f"groq-key-{key_idx+1}-{masked_key}",
                    "performance_index": perf_info["index"],
                    "performance_tier": perf_info["tier"],
                    "parameter_class": perf_info.get("parameter_class", "General"),
                    "multimodal": False,
                }
            })

    # 2. Google Gemini Multimodal Deployment (Vision & Audio)
    google_key = settings.google_api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if google_key:
        gemini_model = settings.gemini_multimodal_model or "gemini/gemini-1.5-flash"
        limits = MODEL_LIMIT_DEFAULTS.get(gemini_model, {"tpm": 1000000, "rpm": 15})
        # Explicit multimodal route
        deployments.append({
            "model_name": "winnie-multimodal-model",
            "litellm_params": {
                "model": gemini_model,
                "api_key": google_key,
                "tpm": limits["tpm"],
                "rpm": limits["rpm"],
            },
            "model_info": {
                "provider": "google",
                "real_model": gemini_model,
                "key_identifier": "google-gemini-key",
                "multimodal": True,
            }
        })

    # 6. Fallback safety if no keys found: use settings directly
    if not deployments:
        deployments.append({
            "model_name": LOGICAL_AGENT_MODEL,
            "litellm_params": {
                "model": settings.model_name,
                "api_base": settings.litellm_base_url or None,
                "api_key": settings.litellm_master_key or None,
            },
            "model_info": {
                "provider": "default",
                "real_model": settings.model_name,
                "key_identifier": "default-fallback",
            }
        })

    return deployments


class MultiProviderRouter:
    """Singleton wrapper around litellm.Router managing the multi-provider token pool."""

    def __init__(self) -> None:
        self._router: Router | None = None
        self._deployments: list[dict[str, Any]] = []
        self.initialize()

    def initialize(self) -> None:
        self._deployments = _build_deployments()
        logger.info(
            f"Initializing MultiProviderRouter with {len(self._deployments)} deployments in token pool."
        )

        try:
            self._router = Router(
                model_list=self._deployments,
                routing_strategy=settings.router_routing_strategy,
                cooldown_time=settings.router_cooldown_time,
                num_retries=0,  # Disabled in LiteLLM: Managed centrally with quota-awareness in MultiProviderRouter
                allowed_fails=2,
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize Router with strategy '{settings.router_routing_strategy}': {exc}. Falling back to 'least-busy'.")
            self._router = Router(
                model_list=self._deployments,
                routing_strategy="least-busy",
                cooldown_time=settings.router_cooldown_time,
                num_retries=0,
            )

    @property
    def router(self) -> Router:
        if self._router is None:
            self.initialize()
        return self._router

    @property
    def is_multimodal_configured(self) -> bool:
        google_key = settings.google_api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return bool(google_key)

    @property
    def multimodal_model_name(self) -> str:
        return settings.gemini_multimodal_model or "gemini/gemini-1.5-flash"

    @property
    def last_dispatched_model(self) -> str:
        return getattr(self, "_last_dispatched_model", settings.model_name)

    @property
    def last_detected_task(self) -> str:
        return getattr(self, "_last_detected_task", "general")

    @property
    def last_routing_events(self) -> list[dict[str, Any]]:
        return getattr(self, "_last_routing_events", [])

    def get_active_cooldowns(self) -> set[str]:
        """Return the set of models currently in cooldown."""
        try:
            if hasattr(self.router, "cooldown_cache") and self.router.cooldown_cache:
                return set(self.router.cooldown_cache.get_active_cooldowns() or [])
        except Exception:
            pass
        return set()

    def completion(self, **kwargs: Any) -> Any:
        """Execute a completion using the task-aware multi-provider token pool."""
        # 0. Enforce FinOps user daily token budget if user_id is provided
        user_id = kwargs.get("user_id")
        if user_id:
            from src.security.finops import finops
            budget_status, used, limit = finops.evaluate_budget(user_id)
            if budget_status == "blocked":
                raise PermissionError(
                    f"Token budget exhausted for user '{user_id}' ({used}/{limit} tokens). "
                    "Request blocked by FinOps spending policy."
                )

        messages = kwargs.get("messages", [])

        # 1. Detect if any message has image/multimodal content
        has_multimodal_content = any(
            isinstance(m.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") in ("image_url", "image")
                for part in m["content"]
            )
            for m in messages if isinstance(m, dict)
        )

        # 2. Extract prompt text for semantic task classification
        prompt_text = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") in ("user", "human"):
                c = m.get("content")
                if isinstance(c, str):
                    prompt_text = c
                elif isinstance(c, list):
                    prompt_text = " ".join(
                        p.get("text", "")
                        for p in c
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                break

        # 3. Classify task intent
        task_type = classify_task_intent(
            prompt=prompt_text,
            tools=kwargs.get("tools"),
            has_images=has_multimodal_content,
        )
        self._last_detected_task = task_type

        # 4. Route based on task type and availability
        if has_multimodal_content:
            if self.is_multimodal_configured:
                kwargs["model"] = "winnie-multimodal-model"
            else:
                # If multimodal key is not set, convert list content to clean text so text-only providers (Groq) don't reject the payload
                sanitized_messages = []
                for m in messages:
                    if isinstance(m, dict) and isinstance(m.get("content"), list):
                        text_parts = [
                            part.get("text", "")
                            for part in m["content"]
                            if isinstance(part, dict) and part.get("type") == "text"
                        ]
                        num_images = sum(
                            1
                            for part in m["content"]
                            if isinstance(part, dict) and part.get("type") in ("image_url", "image")
                        )
                        combined = " ".join(t for t in text_parts if t)
                        if num_images:
                            combined += f"\n\n[User attached {num_images} image(s). Note: Multimodal vision is pending GOOGLE_API_KEY in .env]"
                        sanitized_m = dict(m)
                        sanitized_m["content"] = combined
                        sanitized_messages.append(sanitized_m)
                    else:
                        sanitized_messages.append(m)
                kwargs["messages"] = sanitized_messages
                if "model" not in kwargs or kwargs["model"] in (LOGICAL_AGENT_MODEL, settings.model_name):
                    kwargs["model"] = LOGICAL_AGENT_MODEL
        elif "model" not in kwargs or kwargs["model"] in (LOGICAL_AGENT_MODEL, settings.model_name):
            # Dynamic Task Hierarchy Selection
            task_profile = TASK_PROFILES.get(task_type, TASK_PROFILES["reasoning"])
            active_cooldowns = self.get_active_cooldowns()

            available_models = {
                d["litellm_params"].get("model")
                for d in self._deployments
                if not d.get("model_info", {}).get("multimodal")
            }

            candidate_models = [
                m for m in task_profile["hierarchy"]
                if m in available_models and m not in active_cooldowns
            ]
            if not candidate_models:
                candidate_models = [LOGICAL_AGENT_MODEL]

            from src.agent.error_classifier import (
                error_classifier,
                provider_circuit_breaker,
                ErrorCategory,
                extract_provider_from_model,
            )

            attempt_count = 0
            max_total_attempts = 2  # Hard bounded retry budget across the whole request
            tried_providers: set[str] = set()
            last_error: Exception | None = None
            routing_events: list[dict[str, Any]] = []

            trace_id = kwargs.get("trace_id")
            parent_span_id = kwargs.get("parent_span_id")
            from src.utils.telemetry import get_langfuse_client
            lf = get_langfuse_client()

            for candidate in candidate_models:
                provider = extract_provider_from_model(candidate)
                # If provider circuit breaker is tripped, skip this candidate
                if not provider_circuit_breaker.is_available(provider):
                    logger.info(f"Skipping candidate '{candidate}': provider '{provider}' is currently in cooldown/unhealthy.")
                    routing_events.append({
                        "attempt": attempt_count,
                        "model": candidate,
                        "provider": provider,
                        "status": "skipped",
                        "reason": "circuit_breaker_open",
                    })
                    continue

                # Quota-awareness: if previous candidate on this provider failed due to TPM/rate quota, do not repeat on same quota
                if provider in tried_providers:
                    logger.info(f"Skipping candidate '{candidate}': provider '{provider}' quota was already exhausted in this request.")
                    routing_events.append({
                        "attempt": attempt_count,
                        "model": candidate,
                        "provider": provider,
                        "status": "skipped",
                        "reason": "provider_quota_exhausted",
                    })
                    continue

                if attempt_count >= max_total_attempts:
                    logger.warning(f"Exceeded maximum attempt budget ({max_total_attempts}) for request. Aborting further fallbacks.")
                    break

                attempt_count += 1
                t0_cand = time.perf_counter()
                try:
                    kwargs_call = dict(kwargs)
                    kwargs_call["model"] = candidate
                    self._last_dispatched_model = candidate
                    self._last_detected_task = task_type
                    logger.info(
                        "Dynamic task-based router dispatched request",
                        extra={"task": task_type, "model": candidate, "attempt": attempt_count},
                    )
                    res = self.router.completion(**kwargs_call)
                    provider_circuit_breaker.record_success(provider)
                    dur_ms = round((time.perf_counter() - t0_cand) * 1000, 1)
                    routing_events.append({
                        "attempt": attempt_count,
                        "model": candidate,
                        "provider": provider,
                        "status": "success",
                        "latency_ms": dur_ms,
                    })
                    self._last_routing_events = routing_events

                    if lf and trace_id:
                        try:
                            s = lf.span(
                                trace_id=trace_id,
                                parent_observation_id=parent_span_id,
                                name=f"llm_candidate:{candidate}",
                                input={"attempt": attempt_count, "provider": provider},
                                metadata={"status": "success", "latency_ms": dur_ms},
                            )
                            s.end(output="success")
                        except Exception:
                            pass

                    return res
                except Exception as exc:
                    dur_ms = round((time.perf_counter() - t0_cand) * 1000, 1)
                    last_error = exc
                    classified = error_classifier.classify(exc, model=candidate)
                    provider_circuit_breaker.record_failure(provider, classified)

                    routing_events.append({
                        "attempt": attempt_count,
                        "model": candidate,
                        "provider": provider,
                        "status": "failed",
                        "error_category": classified.category.value,
                        "error_message": str(exc),
                        "latency_ms": dur_ms,
                        "action": classified.recommended_action,
                    })
                    self._last_routing_events = routing_events

                    if lf and trace_id:
                        try:
                            s = lf.span(
                                trace_id=trace_id,
                                parent_observation_id=parent_span_id,
                                name=f"llm_candidate:{candidate}",
                                input={"attempt": attempt_count, "provider": provider},
                                metadata={
                                    "status": "failed",
                                    "error_category": classified.category.value,
                                    "error_message": str(exc),
                                    "latency_ms": dur_ms,
                                    "action": classified.recommended_action,
                                },
                            )
                            s.end(level="ERROR", status_message=f"{classified.category.value}: {exc}")
                        except Exception:
                            pass

                    logger.warning(
                        f"Model candidate '{candidate}' failed [{classified.category.value}] ({exc}). "
                        f"Action: {classified.recommended_action}. Attempt {attempt_count}/{max_total_attempts}."
                    )

                    # Non-retryable errors
                    if classified.category == ErrorCategory.AUTHENTICATION:
                        logger.error(f"Authentication failure on model '{candidate}'. Aborting immediately.")
                        raise

                    if classified.category == ErrorCategory.INVALID_REQUEST:
                        logger.error(f"Invalid request error on model '{candidate}'. Aborting immediately.")
                        raise

                    if classified.category == ErrorCategory.TPM_QUOTA:
                        tried_providers.add(provider)
                        # Immediately compact message payload for any subsequent candidate
                        raw_msgs = kwargs.get("messages", [])
                        if isinstance(raw_msgs, list) and len(raw_msgs) > 2:
                            compacted = []
                            for m in raw_msgs[-3:]:
                                if isinstance(m, dict):
                                    c_copy = dict(m)
                                    c_copy["content"] = str(m.get("content") or "")[:350]
                                    compacted.append(c_copy)
                            kwargs["messages"] = compacted

                    # Mark failed model in cooldown cache if available
                    try:
                        if hasattr(self.router, "cooldown_cache") and self.router.cooldown_cache:
                            self.router.cooldown_cache.add_to_cooldown(candidate)
                    except Exception:
                        pass
                    continue

            if last_error:
                logger.error(f"All model providers failed for task '{task_type}': {last_error}")
                raise RuntimeError(
                    f"All model providers in task hierarchy failed for '{task_type}'. Last error: {last_error}"
                ) from last_error

        if "model" not in kwargs:
            kwargs["model"] = LOGICAL_AGENT_MODEL
        return self.router.completion(**kwargs)

    def get_status(self) -> dict[str, Any]:
        """Return diagnostic metrics about the token pool, task hierarchies, and deployments."""
        total_tpm = sum(d["litellm_params"].get("tpm", 0) for d in self._deployments if d.get("model_name") == LOGICAL_AGENT_MODEL)
        deployment_statuses = []

        active_cooldowns = set()
        try:
            if hasattr(self.router, "cooldown_cache") and self.router.cooldown_cache:
                active_cooldowns = set(self.router.cooldown_cache.get_active_cooldowns() or [])
        except Exception:
            pass

        seen_models = set()
        for idx, d in enumerate(self._deployments):
            params = d["litellm_params"]
            info = d.get("model_info", {})
            model_id = params.get("model")
            
            # De-duplicate entries in diagnostic status (each model is registered under logical & direct alias)
            dedup_key = f"{model_id}-{info.get('key_identifier')}"
            if dedup_key in seen_models:
                continue
            seen_models.add(dedup_key)

            in_cooldown = model_id in active_cooldowns
            is_vision = bool(info.get("multimodal", False))

            deployment_statuses.append({
                "index": len(deployment_statuses) + 1,
                "performance_index": info.get("performance_index"),
                "performance_tier": info.get("performance_tier", "Multimodal Vision" if is_vision else "Unranked"),
                "parameter_class": info.get("parameter_class", "Vision" if is_vision else "General"),
                "is_vision_model": is_vision,
                "provider": info.get("provider", "unknown"),
                "model": model_id,
                "key_label": info.get("key_identifier", "masked"),
                "tpm_limit": params.get("tpm", "unmetered"),
                "rpm_limit": params.get("rpm", "unmetered"),
                "in_cooldown": in_cooldown,
            })

        # Distinct text fallback hierarchy ordered by performance index (excluding vision)
        text_hierarchy = []
        for d in deployment_statuses:
            if not d.get("is_vision_model") and d.get("performance_index") is not None:
                if not any(item["model"] == d["model"] for item in text_hierarchy):
                    text_hierarchy.append({
                        "fallback_order_index": d["performance_index"],
                        "model": d["model"],
                        "performance_tier": d["performance_tier"],
                        "parameter_class": d["parameter_class"],
                        "tpm_limit": d["tpm_limit"],
                    })
        text_hierarchy.sort(key=lambda x: x["fallback_order_index"])

        return {
            "status": "healthy",
            "logical_model": LOGICAL_AGENT_MODEL,
            "routing_strategy": settings.router_routing_strategy,
            "total_deployments": len(deployment_statuses),
            "aggregate_tpm_capacity": total_tpm,
            "last_detected_task": getattr(self, "_last_detected_task", "reasoning"),
            "task_hierarchies": {
                k: {
                    "label": v["label"],
                    "description": v["description"],
                    "primary_model": v["primary_model"],
                    "hierarchy": v["hierarchy"],
                }
                for k, v in TASK_PROFILES.items()
            },
            "text_fallback_hierarchy": text_hierarchy,
            "multimodal_support": {
                "configured": self.is_multimodal_configured,
                "provider": "google_gemini",
                "model": self.multimodal_model_name,
                "api_key_set": bool(settings.google_api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
            },
            "deployments": deployment_statuses,
        }


# Global router singleton
agent_router = MultiProviderRouter()
