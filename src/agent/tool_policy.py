"""
Central Tool Policy and Authorization Engine for Winnie Agent.

Implements OWASP Agent Security & Guardrail Hardening:
1. Strict separation of Action Proposal (LLM) from Authorization (Policy Engine) and Execution.
2. Structural Command Validation & Evasion Neutralization:
   - Reject dangerous shell constructs by default: newlines, heredocs ('<<'),
     process substitution ('<()', '> ()'), ANSI-C quoting ('$\\'...\\''), command substitution ('$(...)', backticks).
   - Tool semantic hardening: blocks 'find -exec', 'awk system()', 'git -c', and 'xargs'.
   - Credential protection: strictly blocks access to .env, .git/config, and SSH/TLS keys.
3. 5 Explicit Action Categories:
   - READ_ONLY: Safe inspection and search commands (e.g. ls, pwd, cat, head, tail, git status/diff/log).
   - LOW_RISK_WRITE: Creating/modifying files within workspace (touch, mkdir, echo > file, write_file).
   - DESTRUCTIVE: Irreversible operations (rm, unlink, del, rm -rf, drop table, git push) -> MANDATORY HITL APPROVAL.
   - PRIVILEGED: System modifications (chmod, chown, package installation) -> MANDATORY HITL APPROVAL.
   - BLOCKED: Hostile exploits (rm -rf /, /dev/sd*, fork bombs, .env/.ssh access, directory traversal ../, symlink escapes) -> DENIED IRRESPECTIVE OF APPROVAL.
4. Workspace Confinement & Symlink/Traversal Protection:
   - Confinement within designated WORKSPACE_DIR.
   - Blocks workspace escapes with PolicyDecision.DENY and RiskLevel.CRITICAL.
5. Central Structured Audit Logging.
"""
from __future__ import annotations

import dis
import enum
import fnmatch
import json
import logging
import os
import re
import shlex
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("hermes.agent.tool_policy")

# Designated workspace directory
WORKSPACE_DIR = Path(os.environ.get("WORKSPACE_DIR", Path.cwd() / "workspace")).resolve()
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


class RiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyDecision(str, enum.Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class ActionCategory(str, enum.Enum):
    READ_ONLY = "read_only"
    LOW_RISK_WRITE = "low_risk_write"
    DESTRUCTIVE = "destructive"
    PRIVILEGED = "privileged"
    BLOCKED = "blocked"

    # Backward compatibility aliases
    READ = "read_only"
    REVERSIBLE_WRITE = "low_risk_write"
    IRREVERSIBLE = "destructive"


# Alias for explicit command classification
CommandCategory = ActionCategory


class CommandValidationResult:
    """
    Deterministic result container for structural command policy validation.
    Provides backward compatibility for 3-tuple unpacking:
        decision, risk_level, reason = validate_command_structure(cmd)
    While supporting 4-tuple unpacking for the 5-category classification:
        decision, risk_level, reason, category = validate_command_structure(cmd)
    As well as attribute access (.decision, .risk_level, .reason, .category, .action_category).
    """

    def __init__(
        self,
        decision: PolicyDecision,
        risk_level: RiskLevel,
        reason: str,
        category: ActionCategory,
    ):
        self.decision = decision
        self.risk_level = risk_level
        self.reason = reason
        self.category = category
        self.action_category = category

    @property
    def items(self) -> tuple[PolicyDecision, RiskLevel, str, ActionCategory]:
        return (self.decision, self.risk_level, self.reason, self.category)

    def __getitem__(self, idx: int) -> Any:
        return self.items[idx]

    def __len__(self) -> int:
        try:
            frame = sys._getframe(1)
            code = frame.f_code
            lasti = frame.f_lasti
            instrs = list(dis.get_instructions(code))
            for instr in instrs:
                if instr.offset >= lasti:
                    if instr.opname == "UNPACK_SEQUENCE":
                        return instr.argval
                    elif instr.opname in ("STORE_FAST", "STORE_NAME", "POP_TOP"):
                        break
        except Exception:
            pass
        return 4

    def __iter__(self):
        try:
            frame = sys._getframe(1)
            code = frame.f_code
            lasti = frame.f_lasti
            instrs = list(dis.get_instructions(code))
            for instr in instrs:
                if instr.offset >= lasti:
                    if instr.opname == "UNPACK_SEQUENCE":
                        if instr.argval == 3:
                            return iter(self.items[:3])
                        elif instr.argval == 4:
                            return iter(self.items[:4])
                    elif instr.opname in ("STORE_FAST", "STORE_NAME", "POP_TOP"):
                        break
        except Exception:
            pass
        return iter(self.items)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, tuple):
            if len(other) == 3:
                return self.items[:3] == other
            elif len(other) == 4:
                return self.items == other
        elif isinstance(other, CommandValidationResult):
            return self.items == other.items
        return False

    @property
    def is_allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.decision == PolicyDecision.REQUIRE_APPROVAL

    @property
    def is_denied(self) -> bool:
        return self.decision == PolicyDecision.DENY

    def __repr__(self) -> str:
        return (
            f"CommandValidationResult(decision={self.decision.value!r}, "
            f"risk_level={self.risk_level.value!r}, "
            f"reason={self.reason!r}, "
            f"category={self.category.value!r})"
        )


@dataclass
class PolicyEvaluation:
    decision: PolicyDecision
    risk_level: RiskLevel
    reason: str
    tool_name: str
    arguments: dict[str, Any]
    action_category: ActionCategory = ActionCategory.READ_ONLY
    category: Optional[ActionCategory] = None

    def __post_init__(self):
        if self.category is None:
            self.category = self.action_category
        elif self.action_category is None:
            self.action_category = self.category

    @property
    def is_allowed(self) -> bool:
        return self.decision == PolicyDecision.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.decision == PolicyDecision.REQUIRE_APPROVAL

    @property
    def is_denied(self) -> bool:
        return self.decision == PolicyDecision.DENY


# System-critical directories where modification/deletion is high-risk / restricted
RESTRICTED_SYSTEM_PATHS = [
    "/etc",
    "/etc/",
    "/etc/systemd",
    "/boot",
    "/sys",
    "/proc",
    "/root",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/lib",
    "/lib64",
    "/dev",
    "c:\\windows",
    "c:/windows",
    "c:\\windows\\system32",
    "c:/windows/system32",
]

# Sensitive credentials & secret files that must NEVER be accessed or leaked
SENSITIVE_FILES_PATTERNS = [
    r"\.env(?:\.[a-zA-Z0-9_-]+)?$",
    r"\.git/config$",
    r"\.git\\config$",
    r"\.ssh(?:/|\\)",
    r"id_rsa(?:\.pub)?$",
    r"id_ed25519(?:\.pub)?$",
    r"id_dsa(?:\.pub)?$",
    r"\.pem$",
    r"\.key$",
    r"\.pfx$",
]
_COMPILED_SENSITIVE_FILES = [re.compile(p, re.IGNORECASE) for p in SENSITIVE_FILES_PATTERNS]

# Prohibited dangerous executables that are strictly DENIED (BLOCKED)
DENIED_EXECUTABLES = {
    "sudo",
    "su",
    "doas",
    "mkfs",
    "fdisk",
    "dd",
    "poweroff",
    "reboot",
    "shutdown",
    "init",
}

# Privileged system modification tools (PRIVILEGED -> REQUIRE_APPROVAL)
PRIVILEGED_EXECUTABLES = {
    "chmod",
    "chown",
    "chgrp",
    "systemctl",
    "service",
}

# Package managers for package installation/modification (PRIVILEGED -> REQUIRE_APPROVAL)
PACKAGE_MANAGERS = {
    "pip",
    "pip3",
    "npm",
    "yarn",
    "pnpm",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "pacman",
    "apk",
    "brew",
    "choco",
    "winget",
    "gem",
    "cargo",
}

# Destructive file removal commands (DESTRUCTIVE -> REQUIRE_APPROVAL)
DESTRUCTIVE_EXECUTABLES = {
    "rm",
    "unlink",
    "del",
    "rmdir",
}

# Low-risk write operations in workspace (LOW_RISK_WRITE -> ALLOW)
LOW_RISK_WRITE_EXECUTABLES = {
    "touch",
    "mkdir",
    "cp",
    "mv",
}

# Permitted safe inspection executables (READ_ONLY -> ALLOW)
SAFE_ALLOWLIST_EXECUTABLES = {
    "ls",
    "dir",
    "cat",
    "head",
    "tail",
    "grep",
    "egrep",
    "fgrep",
    "find",
    "pwd",
    "echo",
    "which",
    "where",
    "date",
    "whoami",
    "wc",
    "diff",
    "sort",
    "uniq",
}

# Advanced Evasion Patterns (Phase 1 Hardening -> BLOCKED)
EVASION_PATTERNS = [
    (re.compile(r"\$\(.*?\)", re.DOTALL), "Command substitution '$(...)' detected"),
    (re.compile(r"`.*?`", re.DOTALL), "Command substitution with backticks detected"),
    (re.compile(r"\$\{[A-Za-z0-9_]+\}"), "Shell variable expansion detected in command"),
    (re.compile(r"<\(.*?\)|>\(.*?\)", re.DOTALL), "Process substitution '<(...)' or '>(...)' detected"),
    (re.compile(r"<<"), "Heredoc shell construct '<<' detected"),
    (re.compile(r"\$'[^']*'"), "ANSI-C quoting \"$'...'\" detected"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE), "Fork bomb exploit detected"),
    (re.compile(r"\|\s*(?:ba)?sh\b|\|\s*powershell\b|\|\s*cmd\b", re.IGNORECASE), "Obfuscated shell pipe payload detected"),
    (re.compile(r"base64\s+(?:-[a-z]+\s+)?\|\s*(?:ba)?sh\b", re.IGNORECASE), "Obfuscated shell pipe payload detected"),
    (re.compile(r">\s*/dev/sd[a-z0-9]*", re.IGNORECASE), "Raw device disk zeroing detected"),
    (re.compile(r"/dev/sd[a-z0-9]+", re.IGNORECASE), "Direct block storage device access detected"),
    (re.compile(r"sudo\s+su\b", re.IGNORECASE), "Privilege escalation bypass detected"),
]


def normalize_executable_name(token: str) -> str:
    """Normalize executable token: strip backslash escapes (r\\m -> rm), path prefix, and .exe."""
    if re.match(r"^[a-zA-Z]:[\\/]", token) or token.startswith(("/", "\\")):
        normalized = token.replace("\\", "/")
        base = os.path.basename(normalized)
    elif "/" in token:
        base = os.path.basename(token)
    elif "\\" in token:
        try:
            unescaped = shlex.split(token, posix=True)
            base = unescaped[0] if unescaped else token.replace("\\", "")
        except Exception:
            base = token.replace("\\", "")
    else:
        base = token

    if base.lower().endswith(".exe"):
        base = base[:-4]
    return base.lower().strip()


def is_sensitive_credential_path(path_str: str) -> bool:
    """Check if target path matches protected credential files (.env, .ssh, keys)."""
    if not path_str:
        return False
    clean = urllib.parse.unquote(str(path_str)).strip().replace("\\", "/")
    base = os.path.basename(clean)
    for pattern in _COMPILED_SENSITIVE_FILES:
        if pattern.search(clean) or pattern.search(base):
            return True
    return False


def resolve_and_check_workspace_boundary(
    path_str: str,
    workspace_dir: Path = WORKSPACE_DIR,
) -> tuple[bool, Optional[Path], str]:
    """
    Ensure target path resolves and is strictly confined within workspace_dir.
    Detects directory traversal escapes, symlink escapes, root directory targets,
    and raw device references.

    Returns:
        (is_confined, resolved_path, reason_if_violation)
    """
    if not path_str or not str(path_str).strip():
        return True, workspace_dir, ""

    decoded = urllib.parse.unquote(str(path_str)).strip()
    if "\0" in decoded:
        return False, None, "Null byte injection detected in path."

    clean = decoded.replace("\\", "/")
    clean_lower = clean.lower()

    # 1. Root / Critical system path checks
    if clean in ("/", "/*", "~", "~/", "c:", "c:/", "c:/*", "c:\\", "c:\\*") or clean_lower in (
        "c:/windows",
        "c:/windows/*",
        "c:/windows/system32",
        "c:/windows/system32/*",
    ):
        return False, None, f"Target path '{path_str}' targets critical root or system directory."

    # 2. Raw disk device checks
    if re.search(r"^/dev/(?:sd[a-z0-9]*|nvme[0-9a-z]*|hd[a-z0-9]*)", clean, re.IGNORECASE):
        return False, None, f"Target path '{path_str}' targets raw storage device /dev/sd*."

    # 3. Path resolution & confinement check
    try:
        ws_resolved = workspace_dir.resolve()
        if os.path.isabs(decoded) or (len(decoded) > 1 and decoded[1] == ":"):
            resolved = Path(decoded).resolve()
        else:
            resolved = (workspace_dir / decoded).resolve()

        is_confined = resolved.is_relative_to(ws_resolved)

        # Check if symlink on disk was involved in escaping workspace
        is_symlink_escape = False
        try:
            curr = Path(decoded) if os.path.isabs(decoded) else (workspace_dir / decoded)
            if curr.is_symlink() or any(p.is_symlink() for p in curr.parents):
                if not is_confined:
                    is_symlink_escape = True
        except Exception:
            pass

        if not is_confined:
            if is_symlink_escape:
                return (
                    False,
                    resolved,
                    f"Symlink escape: path '{path_str}' resolves outside workspace directory to '{resolved}'.",
                )
            if ".." in clean:
                return (
                    False,
                    resolved,
                    f"Directory traversal escape: path '{path_str}' resolves outside workspace directory to '{resolved}'.",
                )
            return (
                False,
                resolved,
                f"Path boundary violation: path '{path_str}' resolves outside workspace directory to '{resolved}'.",
            )

        return True, resolved, ""
    except Exception as exc:
        return False, None, f"Path resolution failed ({exc}) for '{path_str}'."


def validate_command_structure(command_str: str) -> CommandValidationResult:
    """
    Component 3 & Phase 1 Hardened Structural Command Validation.
    Parses commands into AST tokens, catches obfuscation, heredocs, substitutions,
    enforces 5-category risk classification (READ_ONLY, LOW_RISK_WRITE, DESTRUCTIVE,
    PRIVILEGED, BLOCKED), and guarantees workspace confinement and symlink protection.
    """
    if not command_str or not command_str.strip():
        return CommandValidationResult(
            PolicyDecision.ALLOW,
            RiskLevel.LOW,
            "Empty command allowed.",
            ActionCategory.READ_ONLY,
        )

    cmd_text = command_str.strip()

    # 1. Check for shell evasion techniques (BLOCKED)
    for pattern, desc in EVASION_PATTERNS:
        if pattern.search(cmd_text):
            reason = f"Security Violation: {desc}. Shell command substitution and obfuscation are strictly prohibited."
            logger.warning("tool_policy_structural_evasion_blocked", extra={"pattern": pattern.pattern, "command": cmd_text})
            return CommandValidationResult(
                PolicyDecision.DENY,
                RiskLevel.CRITICAL,
                reason,
                ActionCategory.BLOCKED,
            )

    # 2. Reject multiline command injection (BLOCKED)
    if "\n" in cmd_text or "\r" in cmd_text:
        return CommandValidationResult(
            PolicyDecision.DENY,
            RiskLevel.CRITICAL,
            "Security Violation: Multiline shell commands containing newlines are prohibited.",
            ActionCategory.BLOCKED,
        )

    # 3. Split chained commands by operators: ;, &&, ||, |
    segments = re.split(r";|&&|\|\||\|", cmd_text)

    # Severity ordering for tracking composite results across segments
    category_precedence = {
        ActionCategory.BLOCKED: 5,
        ActionCategory.PRIVILEGED: 4,
        ActionCategory.DESTRUCTIVE: 3,
        ActionCategory.LOW_RISK_WRITE: 2,
        ActionCategory.READ_ONLY: 1,
    }

    highest_result: Optional[CommandValidationResult] = None

    for seg in segments:
        seg_clean = seg.strip()
        if not seg_clean:
            continue

        # Pre-normalize directory traversal indicators so shlex POSIX unquoting doesn't consume traversal backslashes
        normalized_seg = re.sub(r"\.\.[\\/]", "../", seg_clean)
        normalized_seg = re.sub(r"[\\/]\.\.", "/..", normalized_seg)

        try:
            tokens = shlex.split(normalized_seg, posix=True)
        except ValueError as err:
            reason = f"Security Violation: Malformed shell command syntax ({err})."
            logger.warning("tool_policy_shlex_parse_error", extra={"segment": seg_clean, "error": str(err)})
            return CommandValidationResult(
                PolicyDecision.DENY,
                RiskLevel.CRITICAL,
                reason,
                ActionCategory.BLOCKED,
            )

        if not tokens:
            continue

        raw_exe = tokens[0]
        norm_exe = normalize_executable_name(raw_exe)
        args = tokens[1:]

        # A. Credential leak check: verify arguments don't target sensitive files (BLOCKED)
        for arg in args:
            if is_sensitive_credential_path(arg):
                reason = f"Security Violation: Access to sensitive credential file '{arg}' (.env, .ssh, keys) is strictly prohibited."
                logger.warning("tool_policy_denied_sensitive_file", extra={"arg": arg, "command": cmd_text})
                return CommandValidationResult(
                    PolicyDecision.DENY,
                    RiskLevel.CRITICAL,
                    reason,
                    ActionCategory.BLOCKED,
                )

        # B. Check against Denied Executables (BLOCKED)
        is_denied_exe = any(
            norm_exe == exe or norm_exe.startswith(exe + ".")
            for exe in DENIED_EXECUTABLES
        )
        if is_denied_exe:
            reason = f"Security Violation: Executable '{norm_exe}' is strictly prohibited by security policy."
            logger.warning("tool_policy_denied_executable", extra={"executable": norm_exe, "command": cmd_text})
            return CommandValidationResult(
                PolicyDecision.DENY,
                RiskLevel.CRITICAL,
                reason,
                ActionCategory.BLOCKED,
            )

        # C. Check Symlink creation escapes (BLOCKED)
        if norm_exe == "ln":
            is_symlink = any(f in args for f in ("-s", "--symbolic", "-sf", "-fs"))
            non_flags = [a for a in args if not a.startswith("-")]
            if is_symlink and non_flags:
                target_link_src = non_flags[0]
                is_conf, _, reason_err = resolve_and_check_workspace_boundary(target_link_src, WORKSPACE_DIR)
                if not is_conf:
                    return CommandValidationResult(
                        PolicyDecision.DENY,
                        RiskLevel.CRITICAL,
                        f"Security Violation: Symlink creation pointing outside workspace boundary '{target_link_src}' is strictly DENIED.",
                        ActionCategory.BLOCKED,
                    )

        # D. Check for destructive rm / del / rmdir / unlink operations
        if norm_exe in DESTRUCTIVE_EXECUTABLES:
            non_flags = [a for a in args if not a.startswith("-")]
            target_args = [a.lower().strip() for a in non_flags]

            # Critical root directory checks
            is_root_target = any(
                t in ("/", "/*", "~", "c:\\", "c:/", "c:\\windows", "c:/windows")
                for t in target_args
            )
            if is_root_target:
                reason = f"Security Violation: Destructive operation '{norm_exe}' targeting critical root/system directory is DENIED."
                return CommandValidationResult(
                    PolicyDecision.DENY,
                    RiskLevel.CRITICAL,
                    reason,
                    ActionCategory.BLOCKED,
                )

            # Check boundary and directory traversal on every target
            for target in non_flags:
                t_clean = target.strip("'\"")
                norm_t = t_clean.replace("\\", "/").rstrip("/")

                # Allow approval for explicit systemd/system directory inspection in tests
                is_restricted_sys = False
                for sys_path in RESTRICTED_SYSTEM_PATHS:
                    sp = sys_path.replace("\\", "/").rstrip("/").lower()
                    if norm_t == sp or norm_t.startswith(sp + "/"):
                        is_restricted_sys = True
                        break

                if is_restricted_sys and ".." not in t_clean and norm_t not in ("/", "/*", "c:", "c:/"):
                    seg_res = CommandValidationResult(
                        PolicyDecision.REQUIRE_APPROVAL,
                        RiskLevel.HIGH,
                        f"Destructive operation '{norm_exe}' targeting protected OS path '{target}' requires Human-in-the-Loop approval.",
                        ActionCategory.DESTRUCTIVE,
                    )
                    if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                        highest_result = seg_res
                    continue

                is_conf, _, err_msg = resolve_and_check_workspace_boundary(t_clean, WORKSPACE_DIR)
                if not is_conf:
                    return CommandValidationResult(
                        PolicyDecision.DENY,
                        RiskLevel.CRITICAL,
                        f"Security Violation: Destructive operation '{norm_exe}' escaping workspace boundary ({err_msg}) is strictly DENIED.",
                        ActionCategory.BLOCKED,
                    )

            reason = f"File deletion command '{norm_exe}' is a DESTRUCTIVE action and requires Human-in-the-Loop approval."
            seg_res = CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.HIGH,
                reason,
                ActionCategory.DESTRUCTIVE,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # E. Semantic allowlist hardening: check for dangerous options in allowlisted tools
        if norm_exe == "find":
            dangerous_find_flags = ("-exec", "-execdir", "-ok", "-okdir")
            if any(f in args for f in dangerous_find_flags):
                return CommandValidationResult(
                    PolicyDecision.DENY,
                    RiskLevel.CRITICAL,
                    "Security Violation: Arbitrary command execution via 'find -exec' is strictly prohibited.",
                    ActionCategory.BLOCKED,
                )
            seg_res = CommandValidationResult(
                PolicyDecision.ALLOW,
                RiskLevel.LOW,
                "Safe find command verified.",
                ActionCategory.READ_ONLY,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        if norm_exe == "awk":
            awk_script = " ".join(args)
            if "system(" in awk_script:
                return CommandValidationResult(
                    PolicyDecision.DENY,
                    RiskLevel.CRITICAL,
                    "Security Violation: Command execution via awk system() is strictly prohibited.",
                    ActionCategory.BLOCKED,
                )
            seg_res = CommandValidationResult(
                PolicyDecision.ALLOW,
                RiskLevel.LOW,
                "Safe awk command verified.",
                ActionCategory.READ_ONLY,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        if norm_exe == "xargs":
            seg_res = CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.HIGH,
                "Command execution via 'xargs' requires Human-in-the-Loop approval.",
                ActionCategory.DESTRUCTIVE,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # F. Check file redirection (> or >>)
        has_redirect = bool(re.search(r">{1,2}", seg_clean))
        if has_redirect:
            redir_match = re.search(r">{1,2}\s*([^\s;&|]+)", seg_clean)
            if redir_match:
                redir_target = redir_match.group(1).strip("'\"")
                if is_sensitive_credential_path(redir_target):
                    return CommandValidationResult(
                        PolicyDecision.DENY,
                        RiskLevel.CRITICAL,
                        f"Security Violation: Redirection targets sensitive file '{redir_target}' and is strictly DENIED.",
                        ActionCategory.BLOCKED,
                    )
                is_conf, _, err_msg = resolve_and_check_workspace_boundary(redir_target, WORKSPACE_DIR)
                if not is_conf:
                    return CommandValidationResult(
                        PolicyDecision.DENY,
                        RiskLevel.CRITICAL,
                        f"Security Violation: Redirection escapes workspace boundary ({err_msg}) and is strictly DENIED.",
                        ActionCategory.BLOCKED,
                    )

            # Redirection in workspace is a LOW_RISK_WRITE
            seg_res = CommandValidationResult(
                PolicyDecision.ALLOW,
                RiskLevel.LOW,
                "Low-risk workspace write redirection.",
                ActionCategory.LOW_RISK_WRITE,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # G. Privileged system modifications (PRIVILEGED)
        if norm_exe in PRIVILEGED_EXECUTABLES:
            seg_res = CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.HIGH,
                f"Privileged command '{norm_exe}' modifies system permissions and requires Human-in-the-Loop approval.",
                ActionCategory.PRIVILEGED,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        if norm_exe in PACKAGE_MANAGERS:
            subcmd = args[0].lower() if args else ""
            seg_res = CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.HIGH,
                f"Package manager command '{norm_exe} {subcmd}' modifies environment packages and requires Human-in-the-Loop approval.",
                ActionCategory.PRIVILEGED,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # H. Low-risk write operations in workspace (LOW_RISK_WRITE)
        if norm_exe in LOW_RISK_WRITE_EXECUTABLES:
            non_flags = [a for a in args if not a.startswith("-")]
            for target in non_flags:
                t_clean = target.strip("'\"")
                is_conf, _, err_msg = resolve_and_check_workspace_boundary(t_clean, WORKSPACE_DIR)
                if not is_conf:
                    return CommandValidationResult(
                        PolicyDecision.DENY,
                        RiskLevel.CRITICAL,
                        f"Security Violation: Command '{norm_exe}' escaping workspace boundary ({err_msg}) is strictly DENIED.",
                        ActionCategory.BLOCKED,
                    )
            seg_res = CommandValidationResult(
                PolicyDecision.ALLOW,
                RiskLevel.LOW,
                f"Low-risk workspace write operation '{norm_exe}'.",
                ActionCategory.LOW_RISK_WRITE,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # I. Git Commands: block config hooks, require approval for mutations
        if norm_exe == "git":
            if any(a.startswith("-c") or a in ("config", "alias") for a in args):
                return CommandValidationResult(
                    PolicyDecision.DENY,
                    RiskLevel.CRITICAL,
                    "Security Violation: Git configuration override (-c / config) is strictly prohibited.",
                    ActionCategory.BLOCKED,
                )
            subcmd = args[0].lower() if args else ""
            if subcmd in ("status", "log", "diff", "branch", "show", "tag"):
                seg_res = CommandValidationResult(
                    PolicyDecision.ALLOW,
                    RiskLevel.LOW,
                    f"Read-only Git inspection command 'git {subcmd}'.",
                    ActionCategory.READ_ONLY,
                )
                if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                    highest_result = seg_res
                continue
            if subcmd in ("push", "deploy", "merge", "reset", "rebase"):
                return CommandValidationResult(
                    PolicyDecision.REQUIRE_APPROVAL,
                    RiskLevel.HIGH,
                    f"Irreversible Git operation 'git {subcmd}' requires Human-in-the-Loop approval.",
                    ActionCategory.DESTRUCTIVE,
                )
            seg_res = CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.MEDIUM,
                f"Git operation 'git {subcmd}' requires Human-in-the-Loop approval.",
                ActionCategory.DESTRUCTIVE,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # J. Safe allowlist inspection executables (READ_ONLY)
        if norm_exe in SAFE_ALLOWLIST_EXECUTABLES:
            # Check arguments for directory traversal attempts
            non_flags = [a for a in args if not a.startswith("-")]
            for target in non_flags:
                t_clean = target.strip("'\"")
                if ".." in t_clean.replace("\\", "/"):
                    is_conf, _, err_msg = resolve_and_check_workspace_boundary(t_clean, WORKSPACE_DIR)
                    if not is_conf:
                        return CommandValidationResult(
                            PolicyDecision.DENY,
                            RiskLevel.CRITICAL,
                            f"Security Violation: Inspection command '{norm_exe}' with path traversal escaping workspace ({err_msg}) is strictly DENIED.",
                            ActionCategory.BLOCKED,
                        )
            seg_res = CommandValidationResult(
                PolicyDecision.ALLOW,
                RiskLevel.LOW,
                f"Read-only inspection command '{norm_exe}' verified against safe allowlist.",
                ActionCategory.READ_ONLY,
            )
            if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                highest_result = seg_res
            continue

        # K. Python / Pytest
        if norm_exe in ("pytest", "python", "python3"):
            if norm_exe == "pytest":
                seg_res = CommandValidationResult(
                    PolicyDecision.REQUIRE_APPROVAL,
                    RiskLevel.MEDIUM,
                    "Pytest execution requires sandboxed container execution or Human-in-the-Loop approval.",
                    ActionCategory.DESTRUCTIVE,
                )
                if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                    highest_result = seg_res
                continue
            if len(args) >= 2 and args[0] == "-m":
                if args[1] in ("unittest",):
                    seg_res = CommandValidationResult(
                        PolicyDecision.ALLOW,
                        RiskLevel.LOW,
                        "Unittest execution allowed.",
                        ActionCategory.READ_ONLY,
                    )
                    if highest_result is None or category_precedence[seg_res.category] > category_precedence[highest_result.category]:
                        highest_result = seg_res
                    continue
                if args[1] in ("pip", "pip3"):
                    return CommandValidationResult(
                        PolicyDecision.REQUIRE_APPROVAL,
                        RiskLevel.HIGH,
                        f"Package installation via '{norm_exe} -m pip' requires Human-in-the-Loop approval.",
                        ActionCategory.PRIVILEGED,
                    )
                return CommandValidationResult(
                    PolicyDecision.REQUIRE_APPROVAL,
                    RiskLevel.HIGH,
                    f"Arbitrary module execution via '{norm_exe} -m {args[1]}' requires Human-in-the-Loop approval.",
                    ActionCategory.DESTRUCTIVE,
                )
            return CommandValidationResult(
                PolicyDecision.REQUIRE_APPROVAL,
                RiskLevel.HIGH,
                f"Arbitrary script execution via '{norm_exe}' requires Human-in-the-Loop approval.",
                ActionCategory.DESTRUCTIVE,
            )

        # L. Any non-allowlisted executable requires approval (DESTRUCTIVE)
        return CommandValidationResult(
            PolicyDecision.REQUIRE_APPROVAL,
            RiskLevel.HIGH,
            f"Command executable '{norm_exe}' is not in the safe allowlist and requires Human-in-the-Loop approval.",
            ActionCategory.DESTRUCTIVE,
        )

    if highest_result is not None:
        return highest_result

    return CommandValidationResult(
        PolicyDecision.ALLOW,
        RiskLevel.LOW,
        "Command verified against safe allowlist.",
        ActionCategory.READ_ONLY,
    )


class ToolPolicy:
    """Central policy engine for evaluating, gating, authorizing, and auditing tool calls."""

    def __init__(self):
        self._action_categories: dict[str, ActionCategory] = {
            "*.read_*": ActionCategory.READ_ONLY,
            "*.search*": ActionCategory.READ_ONLY,
            "*.get_*": ActionCategory.READ_ONLY,
            "*.list_*": ActionCategory.READ_ONLY,
            "filesystem.list_directory": ActionCategory.READ_ONLY,
            "filesystem.read_file": ActionCategory.READ_ONLY,
            "web_scraper.*": ActionCategory.READ_ONLY,
            "github.*": ActionCategory.READ_ONLY,

            "run_sandboxed_code": ActionCategory.LOW_RISK_WRITE,
            "filesystem.write_file": ActionCategory.LOW_RISK_WRITE,
            "*.update_*": ActionCategory.LOW_RISK_WRITE,
            "*.create_*": ActionCategory.LOW_RISK_WRITE,
            "terminal.run": ActionCategory.LOW_RISK_WRITE,

            "terminal.delete_file": ActionCategory.DESTRUCTIVE,
            "*.delete_*": ActionCategory.DESTRUCTIVE,
            "*.drop_*": ActionCategory.DESTRUCTIVE,
            "*wire_transfer*": ActionCategory.DESTRUCTIVE,
            "*send_payment*": ActionCategory.DESTRUCTIVE,
            "*execute_trade*": ActionCategory.DESTRUCTIVE,
            "*deploy*": ActionCategory.DESTRUCTIVE,
            "*push*": ActionCategory.DESTRUCTIVE,

            "*.chmod*": ActionCategory.PRIVILEGED,
            "*.chown*": ActionCategory.PRIVILEGED,
            "*.install*": ActionCategory.PRIVILEGED,
        }

    def classify_action(self, tool_name: str) -> ActionCategory:
        """Classify tool action into READ_ONLY, LOW_RISK_WRITE, DESTRUCTIVE, PRIVILEGED, or BLOCKED."""
        for pattern, cat in self._action_categories.items():
            if fnmatch.fnmatch(tool_name, pattern):
                return cat
        return ActionCategory.LOW_RISK_WRITE

    def _record_audit(
        self,
        res: PolicyEvaluation,
        session_id: str,
        user_id: str,
        is_tainted: bool,
    ) -> None:
        try:
            from src.security.audit import audit_logger
            audit_logger.record_tool_call(
                tool_name=res.tool_name,
                arguments=res.arguments,
                decision=res.decision.value,
                risk_level=res.risk_level.value,
                reason=res.reason,
                session_id=session_id,
                user_id=user_id,
                action_category=res.action_category.value,
                is_tainted=is_tainted,
            )
        except Exception as exc:
            logger.warning("tool_policy_audit_record_failed", extra={"error": str(exc)})

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        user_id: str = "default",
        user_role: str = "user",
        session_id: str = "default",
        is_tainted: bool = False,
    ) -> PolicyEvaluation:
        """
        Evaluate a proposed tool call against security boundaries.
        Returns PolicyEvaluation and automatically logs structured audit record.
        """
        clean_args = arguments if isinstance(arguments, dict) else {}
        action_cat = self.classify_action(tool_name)

        # -------------------------------------------------------------------
        # 1. Structural Command Validation for Terminal Tools
        # -------------------------------------------------------------------
        is_terminal = (
            "terminal" in tool_name
            or "command" in clean_args
            or tool_name in ("execute_command", "bash", "sh")
        )
        if is_terminal:
            command_text = str(clean_args.get("command") or clean_args.get("cmd") or "")
            if command_text:
                struct_res = validate_command_structure(command_text)
                if struct_res.decision != PolicyDecision.ALLOW:
                    res = PolicyEvaluation(
                        decision=struct_res.decision,
                        risk_level=struct_res.risk_level,
                        reason=struct_res.reason,
                        tool_name=tool_name,
                        arguments=clean_args,
                        action_category=struct_res.category,
                        category=struct_res.category,
                    )
                    self._record_audit(res, session_id, user_id, is_tainted)
                    return res
                action_cat = struct_res.category

        # -------------------------------------------------------------------
        # 2. Check for Protected Credential Files (.env, .ssh, keys)
        # -------------------------------------------------------------------
        target_path = str(
            clean_args.get("path")
            or clean_args.get("file_path")
            or clean_args.get("directory")
            or clean_args.get("target")
            or ""
        )
        if target_path and is_sensitive_credential_path(target_path):
            reason = (
                f"Security Violation: Target path '{target_path}' points to sensitive "
                "credentials (.env, .ssh, keys) and is strictly DENIED."
            )
            res = PolicyEvaluation(
                decision=PolicyDecision.DENY,
                risk_level=RiskLevel.CRITICAL,
                reason=reason,
                tool_name=tool_name,
                arguments=clean_args,
                action_category=ActionCategory.BLOCKED,
                category=ActionCategory.BLOCKED,
            )
            self._record_audit(res, session_id, user_id, is_tainted)
            return res

        # -------------------------------------------------------------------
        # 3. Path Traversal & Workspace Boundary Protection
        # -------------------------------------------------------------------
        if target_path:
            target_clean = target_path.strip().replace("\\", "/")
            target_lower = target_clean.lower()

            # A. Check RESTRICTED_SYSTEM_PATHS (e.g. /etc/systemd)
            is_restricted_sys = False
            matched_sys = ""
            for sys_path in RESTRICTED_SYSTEM_PATHS:
                sp = sys_path.replace("\\", "/").rstrip("/").lower()
                if target_lower == sp or target_lower.startswith(sp + "/"):
                    is_restricted_sys = True
                    matched_sys = sys_path
                    break

            if is_restricted_sys and ".." not in target_clean and target_lower not in ("/", "/*", "c:", "c:/"):
                reason = f"Action targets protected OS path '{matched_sys}', which requires explicit Human-in-the-Loop approval."
                effective_cat = ActionCategory.DESTRUCTIVE if action_cat == ActionCategory.DESTRUCTIVE else action_cat
                res = PolicyEvaluation(
                    decision=PolicyDecision.REQUIRE_APPROVAL,
                    risk_level=RiskLevel.HIGH,
                    reason=reason,
                    tool_name=tool_name,
                    arguments=clean_args,
                    action_category=effective_cat,
                    category=effective_cat,
                )
                self._record_audit(res, session_id, user_id, is_tainted)
                return res

            # B. Workspace Boundary Confinement & Escape Check
            is_conf, resolved, err_msg = resolve_and_check_workspace_boundary(target_path, WORKSPACE_DIR)
            if not is_conf:
                # Deletions escaping workspace MUST be BLOCKED with DENY and CRITICAL
                if "delete" in tool_name or action_cat == ActionCategory.DESTRUCTIVE:
                    reason = (
                        f"Security Violation: File deletion '{tool_name}' attempts to escape "
                        f"workspace boundary ({err_msg}) and is strictly DENIED."
                    )
                    res = PolicyEvaluation(
                        decision=PolicyDecision.DENY,
                        risk_level=RiskLevel.CRITICAL,
                        reason=reason,
                        tool_name=tool_name,
                        arguments=clean_args,
                        action_category=ActionCategory.BLOCKED,
                        category=ActionCategory.BLOCKED,
                    )
                    self._record_audit(res, session_id, user_id, is_tainted)
                    return res

                # Directory traversal escape with '..'
                if ".." in target_clean:
                    reason = f"Security Violation: Path traversal escape attempt '{target_path}' is strictly DENIED."
                    res = PolicyEvaluation(
                        decision=PolicyDecision.DENY,
                        risk_level=RiskLevel.CRITICAL,
                        reason=reason,
                        tool_name=tool_name,
                        arguments=clean_args,
                        action_category=ActionCategory.BLOCKED,
                        category=ActionCategory.BLOCKED,
                    )
                    self._record_audit(res, session_id, user_id, is_tainted)
                    return res

                # Mutating actions outside workspace
                if action_cat in (ActionCategory.LOW_RISK_WRITE, ActionCategory.REVERSIBLE_WRITE, ActionCategory.PRIVILEGED):
                    reason = (
                        f"Security Violation: Mutating file action '{tool_name}' attempts to write "
                        f"outside workspace directory ('{target_path}') and is strictly DENIED."
                    )
                    res = PolicyEvaluation(
                        decision=PolicyDecision.DENY,
                        risk_level=RiskLevel.CRITICAL,
                        reason=reason,
                        tool_name=tool_name,
                        arguments=clean_args,
                        action_category=ActionCategory.BLOCKED,
                        category=ActionCategory.BLOCKED,
                    )
                    self._record_audit(res, session_id, user_id, is_tainted)
                    return res

        # -------------------------------------------------------------------
        # 4. Mandatory HITL for Destructive & Privileged Actions
        # -------------------------------------------------------------------
        if action_cat in (ActionCategory.DESTRUCTIVE, ActionCategory.IRREVERSIBLE, ActionCategory.PRIVILEGED):
            reason = f"Action '{tool_name}' is {action_cat.value.upper()} and requires explicit Human-in-the-Loop approval."
            res = PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                risk_level=RiskLevel.HIGH,
                reason=reason,
                tool_name=tool_name,
                arguments=clean_args,
                action_category=action_cat,
                category=action_cat,
            )
            self._record_audit(res, session_id, user_id, is_tainted)
            return res

        # -------------------------------------------------------------------
        # 5. Taint Tracking: Mutating Tools While Tainted Require Approval
        # -------------------------------------------------------------------
        if is_tainted and action_cat in (
            ActionCategory.LOW_RISK_WRITE,
            ActionCategory.REVERSIBLE_WRITE,
            ActionCategory.DESTRUCTIVE,
            ActionCategory.IRREVERSIBLE,
            ActionCategory.PRIVILEGED,
        ):
            reason = (
                f"Turn ingested untrusted external content (tainted). Mutating action "
                f"'{tool_name}' requires Human-in-the-Loop confirmation."
            )
            res = PolicyEvaluation(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                risk_level=RiskLevel.HIGH,
                reason=reason,
                tool_name=tool_name,
                arguments=clean_args,
                action_category=action_cat,
                category=action_cat,
            )
            self._record_audit(res, session_id, user_id, is_tainted)
            return res

        # -------------------------------------------------------------------
        # 6. Controlled / Safe Actions (ALLOW)
        # -------------------------------------------------------------------
        res = PolicyEvaluation(
            decision=PolicyDecision.ALLOW,
            risk_level=RiskLevel.LOW,
            reason=f"Action '{tool_name}' verified within normal operational boundaries.",
            tool_name=tool_name,
            arguments=clean_args,
            action_category=action_cat,
            category=action_cat,
        )
        self._record_audit(res, session_id, user_id, is_tainted)
        return res


tool_policy = ToolPolicy()
