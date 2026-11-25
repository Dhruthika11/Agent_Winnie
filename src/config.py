"""
Centralized configuration for the Hermes-like agent platform.
All values are loaded from environment variables (see .env.example).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Multi-Provider Router & Model Pool ---
    # Primary model (the preferred model attempted first by the router)
    model_name: str = "groq/qwen/qwen3.8-27b"
    # Secondary models (used in exact performance order if primary hits 429 rate limits or cooldown)
    fallback_models: str = "groq/openai/gpt-oss-120b,groq/openai/gpt-oss-20b"
    groq_api_keys: str = ""
    litellm_base_url: str = ""
    litellm_master_key: str = ""

    # --- Google Gemini Multimodal (Vision & Audio) ---
    google_api_key: str = ""
    gemini_api_key: str = ""
    gemini_multimodal_model: str = "gemini/gemini-1.5-flash"

    router_routing_strategy: str = "usage-based-routing-v2"
    router_cooldown_time: int = 30
    router_num_retries: int = 3

    # --- Agent loop control ---
    max_agent_iterations: int = 8
    hitl_approval_timeout_seconds: int = 300

    # --- Memory (Mem0) ---
    mem0_api_key: str = ""
    mem0_vector_store: str = "qdrant"
    mem0_vector_store_url: str = "http://localhost:6334"
    mem0_llm_model: str = "groq/openai/gpt-oss-20b"
    mem0_embedder_model: str = "BAAI/bge-small-en-v1.5"
    mem0_embedder_dims: int = 384

    # --- Datastores ---
    redis_url: str = "redis://localhost:6379/0"
    postgres_url: str = "postgresql://hermes:hermes@localhost:5432/hermes_agent"

    # --- Observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    service_name: str = "hermes-agent-platform"

    # --- Guardrails ---
    guardrails_api_key: str = ""

    # --- Sandbox Hardening & Kernel Confinement ---
    sandbox_image: str = "python:3.11-slim"
    sandbox_timeout_seconds: int = 20
    sandbox_network_disabled: bool = True
    sandbox_runtime: str = ""  # Set to "runsc" to engage gVisor sandbox
    sandbox_mcp_url: str = "http://127.0.0.1:8005/sse"

    # --- Authentication & RBAC ---
    auth_enabled: bool = False
    jwt_secret_key: str = "hermes-insecure-dev-secret-key-change-in-prod"
    jwt_algorithm: str = "HS256"
    api_keys: str = ""  # Comma-separated API keys or "key:role" (e.g. "sk-admin-123:admin,sk-op-456:operator")

    # --- Rate Limiting (Token Bucket / Sliding Window) ---
    rate_limit_enabled: bool = False
    rate_limit_rpm: int = 60

    # --- Enterprise Secrets Management ---
    secret_backend: str = "env"  # "env" | "aws_secrets_manager" | "vault" | "gcp_secret_manager"
    vault_addr: str = ""
    vault_token: str = ""
    aws_secret_name: str = ""
    aws_region: str = "us-east-1"

    # --- Distributed Checkpointer ---
    checkpointer_backend: str = "auto"  # "postgres" | "redis" | "memory" | "auto"

    # --- FinOps Budgeting & Spending Caps ---
    max_daily_tokens_per_user: int = 50000
    finops_budget_degradation_pct: float = 0.80
    finops_budget_block_pct: float = 1.00

    # --- Adaptive 3-Tier Model Routing (Platform Configured Models) ---
    tier1_model: str = "groq/openai/gpt-oss-20b"
    tier2_reasoning_model: str = "groq/openai/gpt-oss-120b"
    tier2_coding_model: str = "groq/qwen/qwen3.8-27b"
    tier3_verifier_model: str = "gemini/gemini-1.5-flash"


settings = Settings()
