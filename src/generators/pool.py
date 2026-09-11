"""
GeneratorPool — the fallback-chain orchestrator.

This is the engine behind the "multiple APIs with fallback" requirement:
for every clip, providers are tried top-to-bottom (read from
config/providers.yaml) until one succeeds. Quota / content-policy / transient
errors each trigger a fall-through to the next provider.

Two cross-cutting guards live here:
  * **Budget guard** — paid providers are skipped entirely if the running
    monthly spend would exceed `settings.budget_cap_usd`.
  * **Credit ledger** — free providers are skipped if their free-credit
    counter for the current period is at zero.

All attempts (success or failure) are recorded in the SQLite ledger so you
can audit where each clip came from. Progress events are also emitted to the
optional `EventBus` so the dashboard can show live per-attempt status.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from src.config import Settings, get_providers_config, get_settings
from src.generators.base import (
    ContentRejected,
    GenerationResult,
    QuotaExceeded,
    VideoGenerator,
)

# Concrete providers import `httpx`. Imported lazily inside _default_factory
# so the pool module stays importable on a bare interpreter (unit tests with
# fakes inject their own factory and never touch this path).
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from src.dashboard.accounts import Account, AccountStore
    from src.store import Store

log = logging.getLogger(__name__)

# Default free-credit caps per provider per period. These are the documented
# free-tier allowances; tune in config/providers.yaml if a provider changes.
# Looked up by canonical `provider_key` (e.g. "hailuo"), NOT the per-account
# label — the cap is a property of the provider's free tier, shared across all
# accounts under that provider.
_DEFAULT_CAPS = {
    "local": {"kind": "daily", "cap": 1_000_000},  # essentially infinite for local/free VPS
}


def _credit_spec(providers_config: dict, provider_key: str) -> tuple[str, int]:
    """(kind, cap) for a provider.

    Precedence: settings.yaml `credits:` (dashboard-editable) →
    providers.yaml `credits:` → built-in default.
    """
    try:
        from src.config import get_settings
        settings_cred = (get_settings().get("credits", {}) or {}).get(provider_key) or {}
    except Exception:
        settings_cred = {}
    spec = (providers_config.get("providers", {}) or {}).get(provider_key, {}) or {}
    file_cred = spec.get("credits") or {}
    default = _DEFAULT_CAPS.get(provider_key, {})
    cred = dict(default)
    cred.update({k: v for k, v in file_cred.items() if v is not None})
    cred.update({k: v for k, v in settings_cred.items() if v is not None})
    kind = str(cred.get("kind", "daily"))
    try:
        cap = int(cred.get("cap", 1_000_000))
    except (TypeError, ValueError):
        cap = 1_000_000
    return kind, cap


@dataclass
class BudgetExceeded(Exception):
    """Raised when no clip could be produced without breaching the budget."""
    message: str = ""

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.message or "budget exceeded"


@dataclass
class GenerationFailed(Exception):
    """Raised when one or more generators were attempted but all failed."""
    message: str = ""

    def __str__(self) -> str:  # pragma: no cover — trivial
        return self.message or "generation failed"


class _EventBusLike(Protocol):
    """Structural type — accepts EventBus, NullEventBus, or any duck-typed obj."""
    def provider_try(self, *a, **kw) -> int: ...
    def provider_skip(self, *a, **kw) -> None: ...
    def provider_fail(self, *a, **kw) -> None: ...
    def provider_ok(self, *a, **kw) -> None: ...


@dataclass
class _AccountBoundGenerator:
    """
    Wraps a concrete generator with the (provider, account) it belongs to.

    When the dashboard manages multiple accounts per provider, the chain is a
    list of these — one per (provider, account) — so the pool can stamp the
    correct `account_id` into events/ledger and the dashboard can show
    "hailuo:main" vs "hailuo:backup" attempts.

    `display_name` is what shows up in events and logs. For the legacy
    env-var path it equals just the provider_key (e.g. "hailuo"), preserving
    pre-dashboard behaviour; for dashboard accounts it's "provider:label".
    """
    gen: VideoGenerator
    provider_key: str            # canonical provider name, e.g. "hailuo"
    account_id: int | None       # None = env-var fallback
    label: str                   # "" for env path, else the account label

    @property
    def name(self) -> str:
        # Env-var path: keep the bare provider key so existing tests/log lines
        # that expect "hailuo" continue to work unchanged.
        return self.provider_key if not self.label else f"{self.provider_key}:{self.label}"

    @property
    def is_configured(self) -> bool:
        return self.gen.is_configured

    @property
    def is_free(self) -> bool:
        return self.gen.is_free

    @property
    def cost_per_clip_usd(self) -> float:
        return self.gen.cost_per_clip_usd

    @property
    def watermark_free(self) -> bool:
        return self.gen.watermark_free

    def generate(self, prompt: str, duration_sec: int, out_path: Path,
                 scene_index: int = 0) -> GenerationResult:
        return self.gen.generate(prompt, duration_sec, out_path, scene_index=scene_index)


class GeneratorPool:
    """Owns the ordered list of providers and the budget + credit guards."""

    def __init__(
        self,
        store: "Store",
        settings: Settings | None = None,
        providers_config: dict | None = None,
        factory: Callable[[str, dict, Settings], VideoGenerator] | None = None,
        *,
        account_store: "AccountStore | None" = None,
        event_bus: "_EventBusLike | None" = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.cfg = providers_config or get_providers_config()
        self._factory = factory or _default_factory
        self._custom_factory = factory is not None
        self.account_store = account_store
        self.event_bus = event_bus
        self._generator_cache: dict[str | int, VideoGenerator] = {}
        self._providers: list[_AccountBoundGenerator] = self._build_chain()

    # ------------------------------------------------------------- construction
    def _build_chain(self) -> list[_AccountBoundGenerator]:
        """
        Build the fallback chain, expanding per-account when the dashboard
        has registered multiple accounts for a provider.

        Skip reasons (chain just gets shorter, never raises):
          * `enabled: false` in providers.yaml
          * factory raises ImportError (optional dep missing)
          * no resolvable credential (gen reports !is_configured)
        All three are intentional degraded behaviour.
        """
        defs = self.cfg.get("providers", {})

        # Test path: a custom factory was injected. Honour it verbatim by
        # wrapping each result so the rest of the pool logic is uniform, but
        # use the provider key as the label-free display name (preserving the
        # pre-dashboard `g.name == "hailuo"` contract the tests assert on).
        if self._custom_factory:
            chain: list[_AccountBoundGenerator] = []
            for name in self.cfg.get("order", []):
                spec = defs.get(name, {})
                if not spec.get("enabled", True):
                    continue
                try:
                    gen = self._factory(name, spec, self.settings)
                except ImportError:
                    continue
                if not gen.is_configured:
                    continue
                chain.append(_AccountBoundGenerator(
                    gen=gen, provider_key=name, account_id=None, label="",
                ))
            log.info("GeneratorPool chain: %s", [g.name for g in chain])
            return chain

        # Production path: dashboard account_store + default factory.
        chain = []
        for name in self.cfg.get("order", []):
            spec = defs.get(name, {})
            if not spec.get("enabled", True):
                log.info("provider %s disabled in config — skipping", name)
                continue
            for entry in self._resolve_accounts(name, spec):
                chain.append(entry)
        log.info("GeneratorPool chain: %s", [g.name for g in chain])
        return chain

    def _resolve_accounts(
        self, name: str, spec: dict,
    ) -> list[_AccountBoundGenerator]:
        """Build one or more chain entries for a provider, honouring the dashboard."""
        # --- Dashboard path: expand per enabled account ----------------------
        if self.account_store is not None:
            accounts = self.account_store.resolve(name)
            if accounts:  # dashboard-managed accounts exist for this provider
                entries: list[_AccountBoundGenerator] = []
                for acct in accounts:
                    cache_key = acct.id
                    cached_gen = self._generator_cache.get(cache_key)
                    # If we already have a cached generator for this account and its URL/key is unchanged, reuse it
                    if cached_gen and getattr(cached_gen, "env_value", None) == acct.api_key:
                        gen = cached_gen
                    else:
                        try:
                            gen = self._factory_for_account(name, spec, acct)
                            self._generator_cache[cache_key] = gen
                        except ImportError as exc:
                            log.warning("provider %s import failed (%s) — skipping account %s",
                                        name, exc, acct.label)
                            continue
                    if not gen.is_configured:
                        continue
                    entries.append(_AccountBoundGenerator(
                        gen=gen, provider_key=name,
                        account_id=acct.id, label=acct.label,
                    ))
                return entries
            # No dashboard accounts → fall through to env-var path below.

        # --- Env-var path (legacy + when dashboard has no accounts) ----------
        cache_key = f"env_{name}"
        cached_gen = self._generator_cache.get(cache_key)
        if cached_gen:
            gen = cached_gen
        else:
            try:
                gen = self._factory(name, spec, self.settings)
                self._generator_cache[cache_key] = gen
            except ImportError as exc:
                log.warning("provider %s module import failed (%s) — skipping", name, exc)
                return []
        if not gen.is_configured:
            log.info("provider %s missing credential — skipping", name)
            return []
        return [_AccountBoundGenerator(
            gen=gen, provider_key=name, account_id=None, label="",
        )]

    def _factory_for_account(
        self, name: str, spec: dict, account: "Account",
    ) -> VideoGenerator:
        """
        Build a generator using dashboard-supplied credentials.

        Mirrors `_default_factory` but pulls the server URL from the
        Account object instead of `settings.env(...)`.
        """
        from src.generators.colab import ColabGenerator
        from src.generators.local import LocalGenerator

        if name == "colab":
            # A "colab" dashboard account stores the tunnel URL as its key.
            return ColabGenerator(tunnel_url=account.api_key)
        if name == "local":
            return LocalGenerator(server_url=account.api_key)
        raise ValueError(f"unknown provider in providers.yaml: {name!r}")

    @property
    def chain(self) -> list[_AccountBoundGenerator]:
        return list(self._providers)

    # ------------------------------------------------------------- public API
    def generate(
        self,
        prompt: str,
        *,
        duration_sec: int = 6,
        out_dir: Path | str,
        post_id: int | None = None,
        scene_index: int | None = None,
        total_scenes: int | None = None,
    ) -> GenerationResult:
        """
        Generate one clip, walking the fallback chain.

        Returns the first successful GenerationResult. Raises BudgetExceeded
        if every free provider is exhausted AND no paid provider can run
        within budget. Emits progress events to `event_bus` when one is set.
        """
        # Re-resolve accounts from DB on every clip request so that users can
        # add backup GPU instances dynamically mid-run to manage GPU session timers.
        self._providers = self._build_chain()

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        bus = self.event_bus

        for provider in self._providers:
            if not provider.is_configured:
                log.debug("skip %s (not configured)", provider.name)
                continue

            # --- FREE providers: respect the credit ledger ------------------
            if provider.is_free:
                kind, cap = _credit_spec(self.cfg, provider.provider_key)
                if self.store.credit_remaining(provider.provider_key, kind, cap) <= 0:
                    log.info("skip %s (free credits exhausted for this period)",
                             provider.name)
                    if bus is not None:
                        bus.provider_skip(
                            provider.name, provider.account_id,
                            "free credits exhausted for this period",
                            scene_index=scene_index, total_scenes=total_scenes,
                        )
                    continue
            # --- PAID providers: respect the budget cap ---------------------
            else:
                projected = self.store.monthly_spend_usd() + provider.cost_per_clip_usd
                if projected > self.settings.budget_cap_usd:
                    log.warning(
                        "skip %s — would breach budget cap (%.2f + %.2f > %.2f)",
                        provider.name, self.store.monthly_spend_usd(),
                        provider.cost_per_clip_usd, self.settings.budget_cap_usd,
                    )
                    if bus is not None:
                        bus.provider_skip(
                            provider.name, provider.account_id,
                            "budget cap would be breached",
                            scene_index=scene_index, total_scenes=total_scenes,
                        )
                    continue

            # --- TRY ---------------------------------------------------------
            t0 = time.monotonic()
            if bus is not None:
                bus.provider_try(
                    provider.name, provider.account_id,
                    scene_index=scene_index, total_scenes=total_scenes,
                )
            clip_path = out_dir / f"clip_{uuid.uuid4().hex[:8]}_{provider.name}.mp4"
            try:
                result = provider.generate(prompt, duration_sec, clip_path,
                                           scene_index=scene_index or 0)
            except (QuotaExceeded, ContentRejected) as exc:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                log.info("provider %s rejected clip (%s) — falling through",
                         provider.name, type(exc).__name__)
                self.store.log_generation(
                    provider.provider_key, "failed", post_id=post_id,
                    scene_index=scene_index, error=str(exc),
                )
                self.store.log_api_call(
                    provider=provider.provider_key,
                    status="failed",
                    account_id=provider.account_id,
                    error=str(exc),
                )
                if bus is not None:
                    bus.provider_fail(
                        provider.name, provider.account_id, exc,
                        scene_index=scene_index, total_scenes=total_scenes,
                        elapsed_ms=elapsed_ms,
                    )
                last_error = exc
                continue
            except Exception as exc:  # transient / unknown — try next
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                log.warning("provider %s errored: %s — falling through",
                            provider.name, exc)
                self.store.log_generation(
                    provider.provider_key, "failed", post_id=post_id,
                    scene_index=scene_index, error=str(exc)[:500],
                )
                self.store.log_api_call(
                    provider=provider.provider_key,
                    status="failed",
                    account_id=provider.account_id,
                    error=str(exc)[:500],
                )
                if bus is not None:
                    bus.provider_fail(
                        provider.name, provider.account_id, exc,
                        scene_index=scene_index, total_scenes=total_scenes,
                        elapsed_ms=elapsed_ms,
                    )
                last_error = exc
                continue

            # --- SUCCESS ----------------------------------------------------
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self.store.log_generation(
                provider.provider_key, "success", post_id=post_id,
                scene_index=scene_index, cost_usd=result.cost_usd,
                clip_path=result.clip_path,
            )
            self.store.log_api_call(
                provider=provider.provider_key,
                status="success",
                account_id=provider.account_id,
                cost_usd=result.cost_usd,
            )
            if provider.is_free:
                kind, cap = _credit_spec(self.cfg, provider.provider_key)
                self.store.consume_credit(provider.provider_key, kind, cap=cap)
            if bus is not None:
                bus.provider_ok(
                    provider.name, provider.account_id,
                    scene_index=scene_index, total_scenes=total_scenes,
                    elapsed_ms=elapsed_ms, cost_usd=result.cost_usd,
                    clip_path=str(result.clip_path),
                )
            log.info("clip generated via %s (cost $%.4f) -> %s",
                     provider.name, result.cost_usd, result.clip_path)
            return result

        # Every provider exhausted.
        if last_error is not None:
            raise GenerationFailed(str(last_error))
        raise BudgetExceeded(
            f"no provider produced the clip. last_error={last_error!r} "
            f"monthly_spend=${self.store.monthly_spend_usd():.2f} "
            f"budget_cap=${self.settings.budget_cap_usd:.2f}"
        )


def _default_factory(name: str, spec: dict, settings: Settings) -> VideoGenerator:
    """Map a providers.yaml entry to a concrete generator instance (env-var path).

    The global `gpu.backend` switch decides which GPU family materialises:
    runpod → LocalGenerator, colab → ColabGenerator. The unselected family
    always reports unconfigured so it drops out of the chain untouched.
    """
    from src.generators.colab import ColabGenerator, selected_backend
    from src.generators.local import LocalGenerator

    if name == "colab":
        return ColabGenerator()
    if name == "local":
        if selected_backend() == "colab":
            return LocalGenerator(server_url="")  # unconfigured → skipped
        return LocalGenerator(server_url=settings.env(spec.get("env_key", "")))
    raise ValueError(f"unknown provider in providers.yaml: {name!r}")
