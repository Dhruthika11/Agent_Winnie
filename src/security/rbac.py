"""
Enterprise Role-Based Access Control (RBAC) & Authorization Layer.

Enforces security and permissions strictly OUTSIDE the LLM at the tool execution boundary:
1. Maps authenticated user roles ('viewer', 'operator', 'admin') to granular permissions.
2. Gates tool execution:
   - 'viewer': web search, read public URLs, conversation interaction.
   - 'operator': 'viewer' + sandboxed code execution, task queue submission.
   - 'admin': 'operator' + MCP management, circuit tripping, deployment approvals, metrics, and audit logs.
3. Enforces resource isolation and audit trail on authorization decisions.
"""
from __future__ import annotations

import enum
import logging
from typing import Any, Optional

logger = logging.getLogger("hermes.security.rbac")


class Role(str, enum.Enum):
    VIEWER = "viewer"
    USER = "viewer"  # Alias
    OPERATOR = "operator"
    ADMIN = "admin"


class Permission(str, enum.Enum):
    # Conversations
    CONVERSATION_READ = "conversation:read"
    CONVERSATION_WRITE = "conversation:write"
    CONVERSATION_DELETE = "conversation:delete"

    # Tools
    TOOL_SEARCH = "tool:search"
    TOOL_READ_URL = "tool:read_url"
    TOOL_SANDBOX_EXECUTE = "tool:sandbox_execute"
    TOOL_CIRCUIT_MANAGE = "tool:circuit_manage"
    TOOL_MCP_MANAGE = "tool:mcp_manage"

    # System & Platform
    SYSTEM_METRICS_READ = "system:metrics_read"
    SYSTEM_DEPLOY_APPROVE = "system:deploy_approve"
    RECOVERY_MANAGE = "recovery:manage"
    TENANT_AUDIT_READ = "tenant:audit_read"


# Direct role-to-permissions mapping (cumulative least-privilege)
ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    Role.VIEWER.value: {
        Permission.CONVERSATION_READ,
        Permission.CONVERSATION_WRITE,
        Permission.TOOL_SEARCH,
        Permission.TOOL_READ_URL,
    },
    Role.OPERATOR.value: {
        Permission.CONVERSATION_READ,
        Permission.CONVERSATION_WRITE,
        Permission.TOOL_SEARCH,
        Permission.TOOL_READ_URL,
        Permission.TOOL_SANDBOX_EXECUTE,
        Permission.SYSTEM_METRICS_READ,
    },
    Role.ADMIN.value: {
        Permission.CONVERSATION_READ,
        Permission.CONVERSATION_WRITE,
        Permission.CONVERSATION_DELETE,
        Permission.TOOL_SEARCH,
        Permission.TOOL_READ_URL,
        Permission.TOOL_SANDBOX_EXECUTE,
        Permission.TOOL_CIRCUIT_MANAGE,
        Permission.TOOL_MCP_MANAGE,
        Permission.SYSTEM_METRICS_READ,
        Permission.SYSTEM_DEPLOY_APPROVE,
        Permission.RECOVERY_MANAGE,
        Permission.TENANT_AUDIT_READ,
    },
}

# Mapping of tool names to required permissions
TOOL_PERMISSION_MAP: dict[str, Permission] = {
    "web_scraper.search": Permission.TOOL_SEARCH,
    "web_scraper.read_url": Permission.TOOL_READ_URL,
    "run_sandboxed_code": Permission.TOOL_SANDBOX_EXECUTE,
    "learning.deploy_improvement": Permission.SYSTEM_DEPLOY_APPROVE,
}


class RBACEngine:
    """Central authorization engine evaluating permissions at execution boundaries."""

    def normalize_role(self, role: Optional[str]) -> str:
        """Normalize role string to canonical enum value with viewer default."""
        if not role:
            return Role.VIEWER.value
        clean = str(role).strip().lower()
        if clean in (Role.ADMIN.value, "administrator"):
            return Role.ADMIN.value
        elif clean in (Role.OPERATOR.value, "ops"):
            return Role.OPERATOR.value
        return Role.VIEWER.value

    def has_permission(self, role: Optional[str], permission: Permission | str) -> bool:
        """Check if a given role possesses a specific permission."""
        norm_role = self.normalize_role(role)
        perm_val = permission if isinstance(permission, Permission) else Permission(str(permission))
        allowed_perms = ROLE_PERMISSIONS.get(norm_role, set())
        return perm_val in allowed_perms

    def authorize_tool_call(
        self,
        role: Optional[str],
        tool_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str]:
        """
        Authorize tool execution based on user role and tool permission requirements.
        Returns (is_authorized, reason).
        """
        norm_role = self.normalize_role(role)
        required_perm = TOOL_PERMISSION_MAP.get(tool_name)

        # Fallback for dynamic/unknown MCP tools: default to OPERATOR/ADMIN unless standard read
        if not required_perm:
            if any(term in tool_name.lower() for term in ["exec", "run", "bash", "write", "delete", "deploy"]):
                required_perm = Permission.TOOL_SANDBOX_EXECUTE
            else:
                required_perm = Permission.TOOL_READ_URL

        if not self.has_permission(norm_role, required_perm):
            reason = (
                f"Role '{norm_role}' lacks permission '{required_perm.value}' required to invoke '{tool_name}'."
            )
            logger.warning("rbac_tool_denied", extra={"role": norm_role, "tool": tool_name, "reason": reason})
            return False, reason

        return True, "Authorized"

    def authorize_resource_access(
        self,
        role: Optional[str],
        resource_owner_id: str,
        requesting_user_id: str,
        resource_tenant_id: str = "default",
        requesting_tenant_id: str = "default",
    ) -> bool:
        """Ensure tenant boundaries and user ownership unless role is admin."""
        # Tenant isolation is absolute
        if resource_tenant_id != requesting_tenant_id:
            logger.warning(
                "tenant_cross_access_blocked",
                extra={"res_tenant": resource_tenant_id, "req_tenant": requesting_tenant_id},
            )
            return False

        norm_role = self.normalize_role(role)
        if norm_role == Role.ADMIN.value:
            return True

        return resource_owner_id == requesting_user_id


rbac_engine = RBACEngine()
rbac_manager = rbac_engine  # Alias for backwards compatibility
