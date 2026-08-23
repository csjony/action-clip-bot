"""
AccountStore — provider-credential resolver backed by the `accounts` table.

The dashboard lets the user add multiple API accounts per provider (e.g.
two Hailuo logins to pool their free credits). `AccountStore` is the single
read-path the GeneratorPool uses to materialise one generator per account.

Design:
  * Falls back to plain env-var credentials when no enabled accounts exist
    for a provider. This keeps the original `.env`-only bootstrap working
    without any dashboard setup — same zero-config behaviour as before.
  * Each resolved account is returned as an `Account` value object so the
    pool can stamp `account_id` into events and the credit ledger.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.store import Store


@dataclass(frozen=True)
class Account:
    """A resolved credential for one provider login."""
    id: int | None            # None = env-var fallback (no dashboard row)
    provider: str
    label: str
    api_key: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_env_fallback(self) -> bool:
        return self.id is None

    @property
    def display_name(self) -> str:
        """Used in events + the chain so the dashboard shows which account ran."""
        return f"{self.provider}:{self.label}"


class AccountStore:
    """Reads + writes the `accounts` table and resolves per-provider credentials."""

    def __init__(self, store: Store) -> None:
        self.store = store

    # ----------------------------------------------------------- read paths
    def resolve(self, provider: str, *, env_value: str = "",
                extra_env: dict[str, str] | None = None) -> list[Account]:
        """
        Return the ordered list of accounts to try for `provider`.

        If the dashboard has any enabled accounts for this provider, those are
        returned in priority order — the env-var credential is ignored.

        If there are NO dashboard accounts for this provider, fall back to a
        single Account built from `env_value` (the legacy .env path). The
        returned list is empty when neither path yields a credential, which
        tells the pool to skip this provider entirely.
        """
        rows = self.store.enabled_accounts(provider)
        if rows:
            return [
                Account(
                    id=r["id"],
                    provider=provider,
                    label=r["label"],
                    api_key=r["api_key"],
                    extra=_parse_extra(r.get("extra_json")),
                )
                for r in rows
            ]
        # Env-var fallback. extra_env carries provider-specific companion keys
        # (e.g. MINIMAX_GROUP_ID, KLING_SECRET_KEY) the caller already resolved.
        if not env_value:
            return []
        return [Account(
            id=None,
            provider=provider,
            label="env",
            api_key=env_value,
            extra=dict(extra_env or {}),
        )]

    def list_all(self, provider: str | None = None) -> list[dict]:
        return self.store.list_accounts(provider)

    def get(self, account_id: int) -> dict | None:
        return self.store.get_account(account_id)

    # ----------------------------------------------------------- write paths
    def add(
        self,
        provider: str,
        label: str,
        api_key: str,
        *,
        email: str | None = None,
        extra: dict[str, Any] | None = None,
        enabled: bool = True,
        priority: int = 100,
    ) -> int:
        extra_json = json.dumps(extra) if extra else None
        return self.store.add_account(
            provider, label, api_key,
            email=email, extra_json=extra_json,
            enabled=enabled, priority=priority,
        )

    def update(
        self, account_id: int, *,
        provider: str | None = None,
        label: str | None = None,
        api_key: str | None = None,
        email: str | None = None,
        extra: dict[str, Any] | None = None,
        enabled: bool | None = None,
        priority: int | None = None,
    ) -> bool:
        extra_json = json.dumps(extra) if extra is not None else None
        return self.store.update_account(
            account_id,
            provider=provider, label=label, api_key=api_key, email=email,
            extra_json=extra_json, enabled=enabled, priority=priority,
        )

    def delete(self, account_id: int) -> bool:
        return self.store.delete_account(account_id)


def _parse_extra(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
