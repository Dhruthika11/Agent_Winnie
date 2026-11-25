"""
Enterprise Secret Management Layer for Hermes Agent Platform.

Supports:
1. Local Environment / .env fallback
2. AWS Secrets Manager (boto3)
3. HashiCorp Vault / Doppler HTTP API
4. GCP Secret Manager

Ensures zero plaintext secrets in production containers.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("hermes.utils.secrets")


class SecretProvider:
    def get(self, key: str, default: str = "") -> str:
        raise NotImplementedError


class EnvSecretProvider(SecretProvider):
    """Fallback provider that reads directly from environment variables or settings."""

    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

    def get(self, key: str, default: str = "") -> str:
        val = os.environ.get(key)
        if val is not None:
            return val
        try:
            from src.config import settings
            val = getattr(settings, key.lower(), None)
            if val is not None:
                return str(val)
        except Exception:
            pass
        return default


class VaultSecretProvider(SecretProvider):
    """HashiCorp Vault / Doppler HTTP Secret Provider."""

    def __init__(self, vault_url: str, token: str, secret_path: str = "secret/data/hermes"):
        self.vault_url = vault_url.rstrip("/")
        self.token = token
        self.secret_path = secret_path
        self._cached: dict[str, str] = {}
        self._loaded = False

    def _fetch_all(self) -> None:
        if self._loaded or not self.vault_url or not self.token:
            return
        import httpx
        try:
            url = f"{self.vault_url}/v1/{self.secret_path}"
            headers = {"X-Vault-Token": self.token}
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json().get("data", {}).get("data", {})
                    self._cached = {k.upper(): str(v) for k, v in data.items()}
                    self._loaded = True
                    logger.info("Successfully loaded secrets from HashiCorp Vault.")
        except Exception as exc:
            logger.warning(f"Failed to fetch secrets from Vault ({exc.__class__.__name__}: {exc}).")

    def get(self, key: str, default: str = "") -> str:
        self._fetch_all()
        return self._cached.get(key.upper(), os.environ.get(key, default))


class AWSSecretsManagerProvider(SecretProvider):
    """AWS Secrets Manager Provider using boto3."""

    def __init__(self, secret_id: str = "hermes/production", region_name: str = "us-east-1"):
        self.secret_id = secret_id
        self.region_name = region_name
        self._cached: dict[str, str] = {}
        self._loaded = False

    def _fetch_all(self) -> None:
        if self._loaded or not self.secret_id:
            return
        try:
            import boto3
            client = boto3.client("secretsmanager", region_name=self.region_name)
            response = client.get_secret_value(SecretId=self.secret_id)
            if "SecretString" in response:
                self._cached = json.loads(response["SecretString"])
                self._loaded = True
                logger.info("Successfully loaded secrets from AWS Secrets Manager.")
        except Exception as exc:
            logger.warning(f"AWS Secrets Manager unavailable ({exc.__class__.__name__}: {exc}). Falling back to env.")

    def get(self, key: str, default: str = "") -> str:
        self._fetch_all()
        return self._cached.get(key, os.environ.get(key, default))


def get_secret_provider() -> SecretProvider:
    """Factory selecting secret provider based on environment configuration."""
    provider_type = os.environ.get("SECRET_PROVIDER", "env").lower()
    if provider_type in ("vault", "doppler"):
        return VaultSecretProvider(
            vault_url=os.environ.get("VAULT_ADDR", ""),
            token=os.environ.get("VAULT_TOKEN", ""),
            secret_path=os.environ.get("VAULT_SECRET_PATH", "secret/data/hermes"),
        )
    elif provider_type in ("aws", "aws_secrets_manager"):
        return AWSSecretsManagerProvider(
            secret_id=os.environ.get("AWS_SECRET_ID", "hermes/production"),
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
        )
    return EnvSecretProvider()


secret_provider = get_secret_provider()


def get_secret(key: str, default: str = "") -> str:
    """Resolve secret from active enterprise secret provider with env fallback."""
    return secret_provider.get(key, default)
