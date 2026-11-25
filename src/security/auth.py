"""
Enterprise Authentication & Role-Based Access Control (RBAC).

Supports:
1. JSON Web Tokens (JWT) signed via HMAC-SHA256.
2. Pre-shared API Keys via X-API-Key header.
3. Role-Based Access Control: 'admin', 'operator', 'viewer'.
4. Graceful development fallback when auth_enabled is False.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import jwt
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger("hermes.security.auth")

security_bearer = HTTPBearer(auto_error=False)


class UserAuth(BaseModel):
    user_id: str
    role: str = "viewer"  # "admin" | "operator" | "viewer"
    tenant_id: str = "default"


def create_access_token(
    user_id: str,
    role: str = "viewer",
    tenant_id: str = "default",
    expires_in_seconds: int = 86400,
) -> str:
    """Generate signed JWT access token for a user with role and tenant claims."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "tenant_id": tenant_id,
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> UserAuth:
    """Validate and decode JWT access token."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return UserAuth(
            user_id=payload.get("sub", "unknown"),
            role=payload.get("role", "viewer"),
            tenant_id=payload.get("tenant_id", "default"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _get_api_key_role_map() -> dict[str, tuple[str, str]]:
    """Parse settings.api_keys configured as 'key:role:tenant' or 'key:role'."""
    mapping = {}
    if not settings.api_keys:
        return mapping

    for entry in settings.api_keys.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) == 3:
            mapping[parts[0]] = (parts[1], parts[2])
        elif len(parts) == 2:
            mapping[parts[0]] = (parts[1], "default")
        else:
            mapping[parts[0]] = ("admin", "default")
    return mapping


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> UserAuth:
    """FastAPI Dependency: Authenticate client using Bearer Token or X-API-Key."""
    # 1. Check X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        key_map = _get_api_key_role_map()
        if api_key in key_map:
            role, tenant = key_map[api_key]
            return UserAuth(user_id=f"apikey_{api_key[:8]}", role=role, tenant_id=tenant)
        if settings.auth_enabled:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key provided in X-API-Key header",
            )

    # 2. Check Bearer Token
    if credentials and credentials.credentials:
        return decode_token(credentials.credentials)

    # 3. If auth_enabled is True and neither valid API key nor token is provided
    if settings.auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 4. Open / Development Mode Fallback
    dev_user = request.headers.get("X-User-Id") or "dev_user"
    dev_tenant = request.headers.get("X-Tenant-Id") or "default"
    return UserAuth(user_id=dev_user, role="admin", tenant_id=dev_tenant)


def require_role(*required_roles: str) -> Callable:
    """Dependency factory that restricts route execution to specific roles."""

    def role_checker(user: UserAuth = Depends(get_current_user)) -> UserAuth:
        if user.role not in required_roles and user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access forbidden: User role '{user.role}' lacks required permissions ({', '.join(required_roles)})",
            )
        return user

    return role_checker
