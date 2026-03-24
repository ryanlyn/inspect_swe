"""Helpers for copying host OAuth/auth state into sandboxed agent runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from inspect_ai.util import SandboxEnvironment

DEFAULT_SANDBOX_USER = "inspect_swe"


async def ensure_sandbox_runtime_user(
    sandbox: SandboxEnvironment,
    requested_user: str | None,
    *,
    cwd: str | None = None,
) -> tuple[str | None, str]:
    """Ensure a non-root sandbox user exists for CLI runtime.

    Returns ``(user, home_dir)``. If ``requested_user`` is not ``None``, it is
    respected and only the home directory is resolved.
    """
    if requested_user is not None:
        home = await _resolve_home_dir(sandbox, requested_user, cwd=cwd)
        return requested_user, home

    script = f"""
set -eu
user="{DEFAULT_SANDBOX_USER}"
if ! id -u "$user" >/dev/null 2>&1; then
  if command -v useradd >/dev/null 2>&1; then
    useradd -m -u 1000 -s /bin/bash "$user" 2>/dev/null || useradd -m -s /bin/bash "$user"
  elif command -v adduser >/dev/null 2>&1; then
    adduser -D -h /home/"$user" "$user" 2>/dev/null || true
  fi
fi
home=$(awk -F: -v user="$user" '$1 == user {{ print $6; exit }}' /etc/passwd)
if [ -z "$home" ]; then
  home=/home/$user
fi
mkdir -p "$home"
chown -R "$user":"$user" "$home" 2>/dev/null || true
printf '%s\\n' "$home"
"""
    result = await sandbox.exec(["sh", "-lc", script], cwd=cwd)
    if not result.success:
        raise RuntimeError(f"Failed to create sandbox runtime user: {result.stderr}")
    return DEFAULT_SANDBOX_USER, result.stdout.strip() or f"/home/{DEFAULT_SANDBOX_USER}"


async def _resolve_home_dir(
    sandbox: SandboxEnvironment,
    user: str,
    *,
    cwd: str | None = None,
) -> str:
    result = await sandbox.exec(
        [
            "sh",
            "-lc",
            f"awk -F: '$1 == \"{user}\" {{ print $6; exit }}' /etc/passwd",
        ],
        cwd=cwd,
    )
    if result.success and result.stdout.strip():
        return result.stdout.strip()
    result = await sandbox.exec(["sh", "-c", "echo $HOME"], user=user, cwd=cwd)
    if result.success and result.stdout.strip():
        return result.stdout.strip()
    return f"/home/{user}"


async def copy_optional_host_file_to_sandbox(
    sandbox: SandboxEnvironment,
    *,
    host_env_var: str,
    sandbox_path: str,
    user: str | None = None,
    cwd: str | None = None,
) -> bool:
    """Copy a host-side auth file into the sandbox if configured.

    The host file path is read from ``host_env_var``. When the env var is unset,
    this function is a no-op and returns ``False``.
    """
    host_path = os.environ.get(host_env_var)
    if not host_path:
        return False

    source = Path(host_path).expanduser()
    if not source.exists():
        raise RuntimeError(
            f"{host_env_var} is set but the file does not exist: {source}"
        )

    parent = str(Path(sandbox_path).parent)
    await sandbox.exec(["mkdir", "-p", parent], user=user, cwd=cwd)
    await sandbox.write_file(sandbox_path, source.read_text(encoding="utf-8"))
    return True


async def copy_optional_host_tree_to_sandbox(
    sandbox: SandboxEnvironment,
    *,
    host_env_var: str,
    sandbox_root: str,
    user: str | None = None,
    cwd: str | None = None,
    exclude_names: Iterable[str] = (),
) -> bool:
    """Recursively copy a host directory tree into the sandbox if configured."""
    host_path = os.environ.get(host_env_var)
    if not host_path:
        return False

    source = Path(host_path).expanduser()
    if not source.exists():
        raise RuntimeError(
            f"{host_env_var} is set but the directory does not exist: {source}"
        )
    if not source.is_dir():
        raise RuntimeError(f"{host_env_var} must point to a directory: {source}")

    excluded = set(exclude_names)
    await sandbox.exec(["mkdir", "-p", sandbox_root], user=user, cwd=cwd)

    for root, dirnames, filenames in os.walk(source):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in excluded
            and not (Path(root) / dirname).is_symlink()
        ]

        root_path = Path(root)
        relative_root = root_path.relative_to(source)
        target_root = Path(sandbox_root) / relative_root
        await sandbox.exec(
            ["mkdir", "-p", str(target_root)],
            user=user,
            cwd=cwd,
        )

        for filename in filenames:
            if filename in excluded:
                continue
            source_file = root_path / filename
            if source_file.is_symlink():
                continue
            target_file = target_root / filename
            await sandbox.write_file(str(target_file), source_file.read_bytes())

    return True


async def make_tree_owned_by_user(
    sandbox: SandboxEnvironment,
    path: str,
    *,
    user: str,
    cwd: str | None = None,
) -> None:
    """Best-effort recursive chown for a sandbox path."""
    await sandbox.exec(
        ["sh", "-lc", f"chown -R {user}:{user} '{path}' 2>/dev/null || true"],
        cwd=cwd,
    )


async def secure_path_for_user(
    sandbox: SandboxEnvironment,
    path: str,
    *,
    user: str,
    recursive: bool = False,
    cwd: str | None = None,
) -> None:
    """Apply conservative permissions to a sandbox path."""
    chmod_flag = "-R " if recursive else ""
    await sandbox.exec(
        ["sh", "-lc", f"chmod {chmod_flag}u+rwX,go-rwx '{path}' 2>/dev/null || true"],
        cwd=cwd,
    )
    await make_tree_owned_by_user(sandbox, path, user=user, cwd=cwd)
