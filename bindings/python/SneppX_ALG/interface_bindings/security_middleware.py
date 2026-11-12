"""Security middleware for inference server — auth, rate limiting, prompt/output filtering."""

import os
import time
import hashlib
import logging
import re
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from .api_key_manager import ApiKeyManager, get_key_manager

# S5 AI-safety backend is imported lazily to avoid a hard dependency at
# import time (it falls back to pure Python when the C lib is absent).
_S5PromptFilter = None
_S5OutputVerifier = None
_FirewallRunner = None


def _get_firewall_runner():
    global _FirewallRunner
    if _FirewallRunner is None:
        from .firewall import FirewallRunner as _FR
        _FirewallRunner = _FR
    return _FirewallRunner


def _get_s5_prompt_filter():
    global _S5PromptFilter
    if _S5PromptFilter is None:
        from .s5_safety import S5PromptFilter as _S5PF
        _S5PromptFilter = _S5PF
    return _S5PromptFilter


def _get_s5_output_verifier():
    global _S5OutputVerifier
    if _S5OutputVerifier is None:
        from .s5_safety import S5OutputVerifier as _S5OV
        _S5OutputVerifier = _S5OV
    return _S5OutputVerifier

logger = logging.getLogger(__name__)


# ===========================================================================
#  Configuration dataclasses
# ===========================================================================


@dataclass
class AuthConfig:
    """Authentication configuration."""
    mode: str = "none"  # "none", "bearer", "api_key"
    api_keys: List[str] = field(default_factory=list)
    jwt_secret: Optional[str] = None
    jwt_issuer: str = "sneppx"
    jwt_audience: str = "sneppx-inference"

    def __post_init__(self):
        if self.mode == "api_key" and not self.api_keys:
            env_keys = os.environ.get("SNEPPX_API_KEYS", "")
            if env_keys:
                self.api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    requests_per_minute: int = 60
    burst_size: int = 10
    burst_window_seconds: float = 10.0


@dataclass
class PromptFilterConfig:
    """Prompt injection / jailbreak detection configuration."""
    enabled: bool = True
    max_token_length: int = 4096
    anomaly_threshold: float = 0.85
    use_s5: bool = False
    patterns: List[str] = field(default_factory=lambda: [
        "ignore previous instructions",
        "ignore all instructions",
        "DAN",
        "do anything now",
        "jailbreak",
        "you are now",
        "pretend to be",
        "override",
        "system prompt",
        "developer mode",
        "as a language model",
        "as an ai",
        "I'm going to",
        "from now on",
    ])


@dataclass
class OutputVerifierConfig:
    """Output verification configuration."""
    enabled: bool = True
    toxicity_threshold: float = 0.8
    use_s5: bool = False
    blocked_topics: List[str] = field(default_factory=lambda: [
        "how to make a bomb",
        "instructions for illegal",
        "child exploitation",
        "self-harm methods",
        "suicide methods",
    ])


@dataclass
class SecurityConfig:
    """Complete security configuration for the inference server."""
    auth: AuthConfig = field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    prompt_filter: PromptFilterConfig = field(default_factory=PromptFilterConfig)
    output_verifier: OutputVerifierConfig = field(default_factory=OutputVerifierConfig)

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Create config from environment variables."""
        auth_mode = os.environ.get("SNEPPX_AUTH_MODE", "none")
        auth = AuthConfig(mode=auth_mode)
        rate_limit = RateLimitConfig(
            enabled=os.environ.get("SNEPPX_RATE_LIMIT", "true").lower() == "true",
            requests_per_minute=int(os.environ.get("SNEPPX_RATE_LIMIT_RPM", "60")),
        )
        prompt_filter = PromptFilterConfig(
            enabled=os.environ.get("SNEPPX_PROMPT_FILTER", "true").lower() == "true",
        )
        output_verifier = OutputVerifierConfig(
            enabled=os.environ.get("SNEPPX_OUTPUT_VERIFIER", "true").lower() == "true",
        )
        return cls(auth=auth, rate_limit=rate_limit,
                    prompt_filter=prompt_filter, output_verifier=output_verifier)


# ===========================================================================
#  Rate Limiter (in-memory sliding window)
# ===========================================================================


class _SlidingWindowCounter:
    """Per-key sliding window counter for rate limiting."""

    def __init__(self, max_requests: int, window_seconds: float):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: List[float] = []

    def allow(self) -> bool:
        now = time.time()
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True

    @property
    def remaining(self) -> int:
        cutoff = time.time() - self._window
        active = sum(1 for t in self._timestamps if t > cutoff)
        return max(0, self._max - active)


# ===========================================================================
#  Prompt Filter
# ===========================================================================


class PromptFilter:
    """Detect prompt injection and jailbreak attempts."""

    def __init__(self, config: PromptFilterConfig):
        self._config = config
        self._compiled: List[Tuple[re.Pattern, str]] = []
        for p in config.patterns:
            try:
                self._compiled.append((re.compile(re.escape(p), re.IGNORECASE), p))
            except re.error:
                pass
        self._s5 = None
        if config.use_s5:
            self._s5 = _get_s5_prompt_filter()()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def scan(self, prompt: str) -> str:
        """Scan prompt. Returns ``"clean"`` or a classification string."""
        if not self._config.enabled:
            return "clean"
        if len(prompt) > self._config.max_token_length * 4:
            return "too_long"
        # S5 backend adds encoded-attack / jailbreak detection.
        if self._s5 is not None and self._s5.scan(prompt) == "blocked":
            return "injection"
        for pattern, raw in self._compiled:
            if pattern.search(prompt):
                return "injection"
        return "clean"

    def sanitize(self, prompt: str) -> str:
        """Remove or replace dangerous content."""
        if not self._config.enabled:
            return prompt
        result = prompt
        for pattern, raw in self._compiled:
            result = pattern.sub("[redacted]", result)
        if self._s5 is not None:
            result = self._s5.sanitize(result)
        if len(result) > self._config.max_token_length * 4:
            result = result[: self._config.max_token_length * 4]
        return result

    def add_pattern(self, pattern: str) -> None:
        """Add a runtime pattern and (if enabled) to the S5 backend."""
        if pattern not in self._config.patterns:
            self._config.patterns.append(pattern)
        try:
            self._compiled.append((re.compile(re.escape(pattern), re.IGNORECASE), pattern))
        except re.error:
            pass
        if self._s5 is not None:
            self._s5.add_pattern(pattern)

    def remove_pattern(self, pattern: str) -> None:
        if pattern in self._config.patterns:
            self._config.patterns.remove(pattern)
        self._compiled = [(p, r) for p, r in self._compiled if r != pattern]
        if self._s5 is not None:
            self._s5.remove_pattern(pattern)


# ===========================================================================
#  Output Verifier
# ===========================================================================


class OutputVerifier:
    """Verify model outputs for policy compliance."""

    def __init__(self, config: OutputVerifierConfig):
        self._config = config
        self._s5 = None
        if config.use_s5:
            self._s5 = _get_s5_output_verifier()()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def check(self, text: str) -> str:
        """Check output. Returns ``"clean"`` or ``"blocked"``."""
        if not self._config.enabled:
            return "clean"
        text_lower = text.lower()
        for topic in self._config.blocked_topics:
            if topic.lower() in text_lower:
                return "blocked"
        # S5 backend adds PII / secret-leak detection.
        if self._s5 is not None and self._s5.check(text) == "blocked":
            return "blocked"
        return "clean"

    def sanitize(self, text: str) -> str:
        """Redact blocked topics from output."""
        if not self._config.enabled:
            return text
        result = text
        for topic in self._config.blocked_topics:
            result = result.replace(topic, "[redacted]")
        if self._s5 is not None:
            result = self._s5.sanitize(result)
        return result

    def add_rule(self, rule: str) -> None:
        if rule not in self._config.blocked_topics:
            self._config.blocked_topics.append(rule)
        if self._s5 is not None:
            self._s5.add_rule(rule)

    def remove_rule(self, rule: str) -> None:
        if rule in self._config.blocked_topics:
            self._config.blocked_topics.remove(rule)
        if self._s5 is not None:
            self._s5.remove_rule(rule)


# ===========================================================================
#  Authenticator
# ===========================================================================


class Authenticator:
    """Authenticate requests using Bearer tokens or API keys.

    Supports both static key sets (CLI) and ApiKeyManager (persistent store).
    """

    def __init__(self, config: AuthConfig, key_manager: Optional[ApiKeyManager] = None):
        self._config = config
        self._key_set: Set[str] = set(config.api_keys)
        self._key_manager = key_manager

    @property
    def enabled(self) -> bool:
        return self._config.mode != "none"

    @property
    def mode(self) -> str:
        return self._config.mode

    def authenticate(self, authorization: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return key info dict if authenticated, ``None`` otherwise.

        If using static keys, returns a simple dict with user_id.
        If using ApiKeyManager, returns the full key info from the store.
        """
        if not self.enabled:
            return {"user_id": "anonymous", "tier": "free", "rate_limit_rpm": 60}
        if not authorization:
            return None
        if self._config.mode == "bearer":
            token = self._extract_bearer(authorization)
            if not token:
                return None
            if self._key_manager and token.startswith("sk-sneppx-"):
                info = self._key_manager.validate_key(token)
                if info:
                    return info
            if token in self._key_set:
                return {"user_id": f"user_{hashlib.sha256(token.encode()).hexdigest()[:16]}",
                        "tier": "static", "rate_limit_rpm": self._config.rate_limit_rpm if hasattr(self._config, 'rate_limit_rpm') else 60}
            return None
        if self._config.mode == "api_key":
            key = authorization.strip()
            if not key:
                return None
            if self._key_manager:
                info = self._key_manager.validate_key(key)
                if info:
                    return info
            if key in self._key_set:
                return {"user_id": f"user_{hashlib.sha256(key.encode()).hexdigest()[:16]}",
                        "tier": "static", "rate_limit_rpm": self._config.rate_limit_rpm if hasattr(self._config, 'rate_limit_rpm') else 60}
            return None
        return None

    @staticmethod
    def _extract_bearer(authorization: str) -> Optional[str]:
        """Extract a Bearer credential using case-insensitive scheme matching."""
        scheme, separator, credential = authorization.partition(" ")
        if not separator or scheme.casefold() != "bearer":
            return None
        token = credential.strip()
        if not token or any(character.isspace() for character in token):
            return None
        return token


# ===========================================================================
#  Security Middleware (Facade)
# ===========================================================================


class SecurityMiddleware:
    """Facade combining authenticator, rate limiter, prompt filter, output verifier.

    Usage in FastAPI::

        security = SecurityMiddleware(config)
        app.state.security = security

        # Before processing:
        user = security.authenticate(request)
        ok, reason = security.check_rate_limit(user)
        result = security.filter_prompt(prompt)

        # After generation:
        result = security.verify_output(text)
"""

    def __init__(self, config: Optional[SecurityConfig] = None,
                 key_manager: Optional[ApiKeyManager] = None):
        self.config = config or SecurityConfig()
        self._key_manager = key_manager
        self.authenticator = Authenticator(self.config.auth, key_manager=key_manager)
        self.prompt_filter = PromptFilter(self.config.prompt_filter)
        self.output_verifier = OutputVerifier(self.config.output_verifier)
        self._rate_limiters: Dict[str, _SlidingWindowCounter] = {}
        self._tier_limits: Dict[str, int] = {
            "free": 60, "pro": 1000, "enterprise": 10000, "static": 60,
        }
        self._firewall = None

    @property
    def key_manager(self) -> Optional[ApiKeyManager]:
        return self._key_manager

    @key_manager.setter
    def key_manager(self, value: Optional[ApiKeyManager]):
        self._key_manager = value
        if value is not None:
            self.authenticator._key_manager = value

    @property
    def firewall(self):
        if self._firewall is None:
            fw_cls = _get_firewall_runner()
            self._firewall = fw_cls()
        return self._firewall

    @firewall.setter
    def firewall(self, value):
        self._firewall = value

    def check_firewall(self, client_ip, method="GET", path="/", headers=None, body=None, query="", ssl_object=None, knock_header=""):
        return self.firewall.check_request(
            client_ip=client_ip,
            method=method,
            path=path,
            headers=headers or {},
            body=body,
            query=query,
            ssl_object=ssl_object,
            knock_header=knock_header,
        )

    def authenticate(self, authorization: Optional[str]) -> Optional[Dict[str, Any]]:
        """Authenticate request. Returns key info dict or ``None``."""
        return self.authenticator.authenticate(authorization)

    def check_rate_limit(self, user_id: str, tier: str = "free") -> Tuple[bool, str]:
        """Check rate limit for user. Returns ``(allowed: bool, reason: str)``."""
        if not self.config.rate_limit.enabled:
            return True, ""
        rpm = self._tier_limits.get(tier, self.config.rate_limit.requests_per_minute)
        if user_id not in self._rate_limiters:
            self._rate_limiters[user_id] = _SlidingWindowCounter(rpm, 60.0)
        limiter = self._rate_limiters[user_id]
        if not limiter.allow():
            return False, f"Rate limit exceeded ({rpm}/minute)"
        return True, ""

    def filter_prompt(self, prompt: str) -> Tuple[str, str]:
        """Filter prompt. Returns ``(status, sanitized_or_original)``.

        Status is ``"clean"``, ``"injection"``, ``"too_long"``, or ``"sanitized"``."""
        if not self.prompt_filter.enabled:
            return "clean", prompt
        status = self.prompt_filter.scan(prompt)
        if status == "clean":
            return "clean", prompt
        sanitized = self.prompt_filter.sanitize(prompt)
        return status, sanitized

    def verify_output(self, text: str) -> Tuple[str, str]:
        """Verify output. Returns ``(status, sanitized_or_original)``."""
        if not self.output_verifier.enabled:
            return "clean", text
        status = self.output_verifier.check(text)
        if status == "clean":
            return "clean", text
        sanitized = self.output_verifier.sanitize(text)
        return status, sanitized

    def auth_required_error(self) -> dict:
        return {"detail": "Authentication required"}

    def rate_limit_error(self, reason: str) -> dict:
        return {"detail": reason}

    def prompt_blocked_error(self) -> dict:
        return {"detail": "Prompt blocked by content filter"}

    def firewall_blocked_error(self, result) -> dict:
        return {"detail": result.reason, "status_code": result.status_code}

    def release_concurrent(self, client_ip: str):
        self.firewall.release_concurrent(client_ip)


__all__ = [
    "AuthConfig",
    "RateLimitConfig",
    "PromptFilterConfig",
    "OutputVerifierConfig",
    "SecurityConfig",
    "SecurityMiddleware",
    "Authenticator",
    "PromptFilter",
    "OutputVerifier",
]
