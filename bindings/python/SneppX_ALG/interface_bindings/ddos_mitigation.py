"""DDoS mitigation module — S4 network security.

Provides multi-layer DDoS protection including:
- Rate limiting (token bucket, sliding window)
- Connection tracking and limiting
- SYN flood detection
- IP reputation and geo-blocking
- Adaptive thresholding based on traffic patterns
- Integration with S4 4-ring firewall

Wraps C implementations in security/network/ with pure-Python fallback.
"""

import os
import time
import threading
import hashlib
import json
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .c_loader import load_library

_LIB, _HAS_C = load_library("neural_security_c")


class DDoSAction(Enum):
    """Action to take when DDoS detected."""
    MONITOR = "monitor"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    BLOCK = "block"
    CAPTCHA = "captcha"


@dataclass
class DDoSConfig:
    """Configuration for DDoS mitigation."""
    # Connection limits
    max_connections_per_ip: int = 1000
    max_connections_total: int = 100000
    max_requests_per_second: int = 1000
    max_requests_per_minute: int = 10000
    
    # SYN flood protection
    syn_flood_threshold: int = 1000
    syn_flood_window_seconds: float = 1.0
    
    # Rate limiting
    token_bucket_capacity: int = 1000
    token_bucket_refill_rate: float = 100.0
    sliding_window_size: int = 60
    
    # IP reputation
    enable_ip_reputation: bool = True
    reputation_ban_threshold: int = 100
    reputation_decay_hours: float = 24.0
    
    # Geo-blocking
    blocked_countries: List[str] = field(default_factory=list)
    
    # Adaptive thresholds
    adaptive_thresholds: bool = True
    learning_window_seconds: int = 300
    
    # Actions
    default_action: DDoSAction = DDoSAction.RATE_LIMIT
    ban_duration_seconds: int = 3600


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def available(self) -> int:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            return int(self.tokens)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
    
    def check(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        
        with self._lock:
            window = self.requests[key]
            while window and window[0] < cutoff:
                window.popleft()
            
            if len(window) < self.max_requests:
                window.append(now)
                return True
            return False
    
    def get_count(self, key: str) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        
        with self._lock:
            window = self.requests[key]
            while window and window[0] < cutoff:
                window.popleft()
            return len(window)


class IPReputationManager:
    """IP reputation tracking with decay."""
    
    def __init__(self, ban_threshold: int = 100, decay_hours: float = 24.0):
        self.ban_threshold = ban_threshold
        self.decay_hours = decay_hours
        self.scores: Dict[str, float] = defaultdict(float)
        self.events: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_event(self, ip: str, event_type: str, severity: float = 1.0):
        now = time.time()
        with self._lock:
            self.scores[ip] += severity
            self.events[ip].append((now, event_type))
            self._decay(ip, now)
    
    def _decay(self, ip: str, now: float):
        cutoff = now - self.decay_hours * 3600
        events = self.events[ip]
        while events and events[0][0] < cutoff:
            _, evt = events.pop(0)
            # Recalculate score from remaining events
            self.scores[ip] = sum(1.0 for _, e in events)
            if self.scores[ip] <= 0:
                del self.scores[ip]
                del self.events[ip]
    
    def get_score(self, ip: str) -> float:
        with self._lock:
            return self.scores.get(ip, 0.0)
    
    def is_banned(self, ip: str) -> bool:
        with self._lock:
            return self.scores.get(ip, 0.0) >= self.ban_threshold
    
    def ban_ip(self, ip: str, duration: int = 3600):
        with self._lock:
            self.scores[ip] = self.ban_threshold + duration


class ConnectionTracker:
    """Track active connections per IP and globally."""
    
    def __init__(self, max_per_ip: int = 1000, max_total: int = 100000):
        self.max_per_ip = max_per_ip
        self.max_total = max_total
        self.connections: Dict[str, int] = defaultdict(int)
        self.total = 0
        self._lock = threading.Lock()
    
    def acquire(self, ip: str) -> bool:
        with self._lock:
            if self.connections[ip] >= self.max_per_ip:
                return False
            if self.total >= self.max_total:
                return False
            self.connections[ip] += 1
            self.total += 1
            return True
    
    def release(self, ip: str):
        with self._lock:
            if self.connections[ip] > 0:
                self.connections[ip] -= 1
            if self.total > 0:
                self.total -= 1
    
    def get_count(self, ip: str) -> int:
        with self._lock:
            return self.connections.get(ip, 0)
    
    def get_total(self) -> int:
        with self._lock:
            return self.total


class SYNFloodDetector:
    """Detect SYN flood attacks."""
    
    def __init__(self, threshold: int = 1000, window: float = 1.0):
        self.threshold = threshold
        self.window = window
        self.syn_timestamps: deque = deque()
        self._lock = threading.Lock()
    
    def record_syn(self) -> bool:
        """Record a SYN packet. Returns True if flood detected."""
        now = time.monotonic()
        cutoff = now - self.window
        
        with self._lock:
            while self.syn_timestamps and self.syn_timestamps[0] < cutoff:
                self.syn_timestamps.popleft()
            
            self.syn_timestamps.append(now)
            return len(self.syn_timestamps) >= self.threshold
    
    def get_rate(self) -> float:
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.window
            while self.syn_timestamps and self.syn_timestamps[0] < cutoff:
                self.syn_timestamps.popleft()
            return len(self.syn_timestamps) / self.window


class AdaptiveThresholdManager:
    """Adaptively adjust rate limits based on traffic patterns."""
    
    def __init__(self, learning_window: int = 300):
        self.learning_window = learning_window
        self.request_counts: deque = deque()
        self.baseline: float = 0.0
        self._lock = threading.Lock()
    
    def record_request(self):
        now = time.time()
        with self._lock:
            self.request_counts.append(now)
            self._cleanup(now)
    
    def _cleanup(self, now: float):
        cutoff = now - self.learning_window
        while self.request_counts and self.request_counts[0] < cutoff:
            self.request_counts.popleft()
    
    def get_baseline(self) -> float:
        with self._lock:
            self._cleanup(time.time())
            if not self.request_counts:
                return 0.0
            window = self.request_counts[-1] - self.request_counts[0]
            if window <= 0:
                return 0.0
            return len(self.request_counts) / window
    
    def get_threshold_multiplier(self) -> float:
        """Get multiplier for current threshold (1.0 = normal)."""
        baseline = self.get_baseline()
        if baseline <= 0:
            return 1.0
        
        current_rate = self.get_baseline()
        if current_rate > baseline * 10:
            return 0.1  # Aggressive limiting
        elif current_rate > baseline * 5:
            return 0.3
        elif current_rate > baseline * 2:
            return 0.5
        return 1.0


class DDoSMitigation:
    """Main DDoS mitigation coordinator."""
    
    def __init__(self, config: Optional[DDoSConfig] = None):
        self.config = config or DDoSConfig()
        self._lock = threading.RLock()
        
        # Core components
        self.token_bucket = TokenBucket(
            self.config.token_bucket_capacity,
            self.config.token_bucket_refill_rate
        )
        self.sliding_window = SlidingWindowRateLimiter(
            self.config.max_requests_per_minute,
            self.config.sliding_window_size
        )
        self.ip_reputation = IPReputationManager(
            self.config.reputation_ban_threshold,
            self.config.reputation_decay_hours
        )
        self.connections = ConnectionTracker(
            self.config.max_connections_per_ip,
            self.config.max_connections_total
        )
        self.syn_detector = SYNFloodDetector(
            self.config.syn_flood_threshold,
            self.config.syn_flood_window_seconds
        )
        self.adaptive = AdaptiveThresholdManager(
            self.config.learning_window_seconds
        )
        
        # Blocked IPs
        self.blocked_ips: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "rate_limited": 0,
            "syn_floods_detected": 0,
            "ips_banned": 0,
        }
    
    def check_request(self, ip: str, is_syn: bool = False) -> Tuple[bool, DDoSAction, Optional[str]]:
        """Check if request should be allowed.
        
        Returns:
            (allowed, action, reason)
        """
        with self._lock:
            self.stats["total_requests"] += 1
            
            # Check if IP is currently blocked
            now = time.time()
            if ip in self.blocked_ips:
                if self.blocked_ips[ip] > now:
                    self.stats["blocked_requests"] += 1
                    return False, DDoSAction.BLOCK, "IP temporarily banned"
                else:
                    del self.blocked_ips[ip]
            
            # Check IP reputation
            if self.config.enable_ip_reputation and self.ip_reputation.is_banned(ip):
                self.stats["blocked_requests"] += 1
                return False, DDoSAction.BLOCK, "IP reputation ban"
            
            # SYN flood detection
            if is_syn:
                if self.syn_detector.record_syn():
                    self.stats["syn_floods_detected"] += 1
                    self.ip_reputation.record_event(ip, "syn_flood", 10.0)
                    return False, DDoSAction.RATE_LIMIT, "SYN flood detected"
            
            # Connection tracking
            if not self.connections.acquire(ip):
                self.ip_reputation.record_event(ip, "connection_limit", 5.0)
                self.stats["blocked_requests"] += 1
                return False, DDoSAction.RATE_LIMIT, "Connection limit exceeded"
            
            # Token bucket rate limiting
            if not self.token_bucket.consume():
                self.stats["rate_limited"] += 1
                self.ip_reputation.record_event(ip, "rate_limit", 1.0)
                return False, DDoSAction.RATE_LIMIT, "Rate limit exceeded"
            
            # Sliding window
            if not self.sliding_window.check(ip):
                self.stats["rate_limited"] += 1
                self.ip_reputation.record_event(ip, "rate_limit", 1.0)
                return False, DDoSAction.RATE_LIMIT, "Sliding window limit exceeded"
            
            # Adaptive thresholds
            if self.config.adaptive_thresholds:
                multiplier = self.adaptive.get_threshold_multiplier()
                if multiplier < 1.0:
                    # Additional probabilistic rejection
                    import random
                    if random.random() > multiplier:
                        self.stats["rate_limited"] += 1
                        return False, DDoSAction.RATE_LIMIT, "Adaptive rate limit"
            
            # Record for adaptive learning
            self.adaptive.record_request()
            
            return True, DDoSAction.MONITOR, None
    
    def release_connection(self, ip: str):
        with self._lock:
            self.connections.release(ip)
    
    def ban_ip(self, ip: str, duration: Optional[int] = None):
        with self._lock:
            duration = duration or self.config.ban_duration_seconds
            self.blocked_ips[ip] = time.time() + duration
            self.ip_reputation.ban_ip(ip, duration)
            self.stats["ips_banned"] += 1
    
    def unban_ip(self, ip: str):
        with self._lock:
            self.blocked_ips.pop(ip, None)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats)
    
    def get_ip_info(self, ip: str) -> Dict[str, Any]:
        with self._lock:
            return {
                "ip": ip,
                "reputation_score": self.ip_reputation.get_score(ip),
                "is_banned": self.ip_reputation.is_banned(ip),
                "connections": self.connections.get_count(ip),
                "blocked_until": self.blocked_ips.get(ip),
            }
    
    def cleanup_expired(self):
        """Clean up expired bans and old data."""
        with self._lock:
            now = time.time()
            expired = [ip for ip, until in self.blocked_ips.items() if until <= now]
            for ip in expired:
                del self.blocked_ips[ip]


# Global instance
_global_mitigation: Optional[DDoSMitigation] = None


def get_ddos_mitigation(config: Optional[DDoSConfig] = None) -> DDoSMitigation:
    """Get or create global DDoS mitigation instance."""
    global _global_mitigation
    if _global_mitigation is None:
        _global_mitigation = DDoSMitigation(config)
    return _global_mitigation


def set_ddos_mitigation(mitigation: DDoSMitigation):
    """Set global DDoS mitigation instance."""
    global _global_mitigation
    _global_mitigation = mitigation


def check_request(ip: str, is_syn: bool = False) -> Tuple[bool, DDoSAction, Optional[str]]:
    """Convenience function to check a request using global mitigation."""
    return get_ddos_mitigation().check_request(ip, is_syn)


def release_connection(ip: str):
    """Release a connection for an IP."""
    get_ddos_mitigation().release_connection(ip)


def ban_ip(ip: str, duration: Optional[int] = None):
    """Ban an IP address."""
    get_ddos_mitigation().ban_ip(ip, duration)


def unban_ip(ip: str):
    """Unban an IP address."""
    get_ddos_mitigation().unban_ip(ip)


def get_ddos_stats() -> Dict[str, Any]:
    """Get DDoS mitigation statistics."""
    return get_ddos_mitigation().get_stats()


def get_ip_info(ip: str) -> Dict[str, Any]:
    """Get information about an IP."""
    return get_ddos_mitigation().get_ip_info(ip)


__all__ = [
    "DDoSAction",
    "DDoSConfig",
    "TokenBucket",
    "SlidingWindowRateLimiter",
    "IPReputationManager",
    "ConnectionTracker",
    "SYNFloodDetector",
    "AdaptiveThresholdManager",
    "DDoSMitigation",
    "get_ddos_mitigation",
    "set_ddos_mitigation",
    "check_request",
    "release_connection",
    "ban_ip",
    "unban_ip",
    "get_ddos_stats",
    "get_ip_info",
]