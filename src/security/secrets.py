"""
Enterprise Secret Management Layer.

Supports fetching secrets on startup from:
- AWS Secrets Manager
- HashiCorp Vault
- Google Cloud Secret Manager
- Environment / .env (default development fallback)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.config import settings

logger = logging.getLogger("hermes.security.secrets")


class SecretsManager:
    def __init__(self):
        self._cache: dict[str, Any] = {}

    def load_secrets(self) -> dict[str, Any]:
        """Fetch secrets from configured cloud vault and populate environment variables."""
        backend = (settings.secret_backend or "env").lower()

        if backend == "aws_secrets_manager":
            return self._load_from_aws()
        elif backend == "vault":
            return self._load_from_vault()
        elif backend == "gcp_secret_manager":
            return self._load_from_gcp()
        else:
            logger.info("using_env_secret_backend")
            return {}

    def _load_from_aws(self) -> dict[str, Any]:
        try:
            import boto3

            logger.info(
                "fetching_secrets_from_aws",
                extra={"secret_name": settings.aws_secret_name, "region": settings.aws_region},
            )
            client = boto3.client("secretsmanager", region_name=settings.aws_region)
            resp = client.get_secret_value(SecretId=settings.aws_secret_name)
            raw = resp.get("SecretString") or "{}"
            secrets = json.loads(raw)
            self._apply_secrets(secrets)
            logger.info("aws_secrets_loaded_successfully", extra={"keys_count": len(secrets)})
            return secrets
        except Exception as exc:
            logger.warning(
                f"[Secrets] AWS Secrets Manager failed ({exc.__class__.__name__}: {exc}). "
                "Retaining local environment variables."
            )
            return {}

    def _load_from_vault(self) -> dict[str, Any]:
        try:
            import hvac

            logger.info("fetching_secrets_from_hashicorp_vault", extra={"addr": settings.vault_addr})
            client = hvac.Client(url=settings.vault_addr, token=settings.vault_token)
            read_resp = client.secrets.kv.v2.read_secret_version(path="hermes")
            secrets = read_resp.get("data", {}).get("data", {})
            self._apply_secrets(secrets)
            logger.info("vault_secrets_loaded_successfully", extra={"keys_count": len(secrets)})
            return secrets
        except Exception as exc:
            logger.warning(
                f"[Secrets] HashiCorp Vault fetch failed ({exc.__class__.__name__}: {exc}). "
                "Retaining local environment variables."
            )
            return {}

    def _load_from_gcp(self) -> dict[str, Any]:
        try:
            from google.cloud import secretmanager

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{os.environ.get('GCP_PROJECT')}/secrets/hermes-secrets/versions/latest"
            response = client.access_secret_version(request={"name": name})
            raw = response.payload.data.decode("UTF-8")
            secrets = json.loads(raw)
            self._apply_secrets(secrets)
            logger.info("gcp_secrets_loaded_successfully", extra={"keys_count": len(secrets)})
            return secrets
        except Exception as exc:
            logger.warning(
                f"[Secrets] GCP Secret Manager failed ({exc.__class__.__name__}: {exc}). "
                "Retaining local environment variables."
            )
            return {}

    def _apply_secrets(self, secrets: dict[str, Any]):
        """Inject fetched secrets into environment and update configuration settings."""
        self._cache.update(secrets)
        for k, v in secrets.items():
            if isinstance(v, str):
                os.environ[k] = v
                # Synchronize settings if matching field exists
                k_lower = k.lower()
                if hasattr(settings, k_lower):
                    try:
                        setattr(settings, k_lower, v)
                    except Exception:
                        pass


secrets_manager = SecretsManager()
