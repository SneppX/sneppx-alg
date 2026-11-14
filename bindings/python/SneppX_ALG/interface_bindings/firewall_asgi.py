"""ASGI integration that enforces firewall policy before application handlers run."""

import json


_SINGLETON_HEADERS = {
    "authorization",
    "content-length",
    "content-type",
    "host",
    "transfer-encoding",
}


class FirewallASGIMiddleware:
    """Bounded-body ASGI firewall middleware for HTTP applications.

    The middleware buffers at most the configured request-body limit, evaluates
    the three firewall rings, then replays the accepted body to the downstream
    ASGI application. Concurrency slots are always released after dispatch.
    """

    def __init__(self, app, security_getter):
        self.app = app
        self.security_getter = security_getter

    @staticmethod
    async def _reject(send, status_code, detail):
        body = json.dumps({"detail": str(detail)}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": int(status_code),
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        try:
            security = self.security_getter()
            if security is None:
                await self.app(scope, receive, send)
                return
            firewall = security.firewall
            if not getattr(getattr(firewall, "config", None), "enabled", True):
                await self.app(scope, receive, send)
                return
            max_body_size = int(getattr(firewall, "max_body_size", 1_048_576))
            max_header_count = int(getattr(firewall, "max_header_count", 100))
        except Exception:
            await self._reject(send, 503, "Firewall is unavailable")
            return

        raw_headers = scope.get("headers", ())
        if len(raw_headers) > max_header_count:
            await self._reject(send, 431, "Too many request headers")
            return

        headers = {}
        for raw_name, raw_value in raw_headers:
            try:
                name = raw_name.decode("ascii").strip().lower()
                value = raw_value.decode("latin-1")
            except (AttributeError, UnicodeDecodeError):
                await self._reject(send, 400, "Malformed request header")
                return
            if name in headers:
                if name in _SINGLETON_HEADERS:
                    await self._reject(send, 400, "Duplicate security-sensitive request header")
                    return
                separator = "; " if name == "cookie" else ", "
                headers[name] = headers[name] + separator + value
            else:
                headers[name] = value

        if "content-length" in headers and "transfer-encoding" in headers:
            await self._reject(send, 400, "Conflicting request framing headers")
            return

        declared_length = headers.get("content-length")
        if declared_length is not None:
            if len(declared_length) > 20 or not declared_length.isdecimal():
                await self._reject(send, 400, "Invalid content-length header")
                return
            if int(declared_length) > max_body_size:
                await self._reject(send, 413, "Request body too large")
                return

        chunks = []
        body_size = 0
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                return
            if message_type != "http.request":
                continue
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                await self._reject(send, 400, "Invalid request body")
                return
            body_size += len(chunk)
            if body_size > max_body_size:
                await self._reject(send, 413, "Request body too large")
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        client = scope.get("client")
        client_ip = client[0] if client and len(client) > 0 else ""
        query_bytes = scope.get("query_string", b"")
        try:
            query = query_bytes.decode("latin-1") if isinstance(query_bytes, bytes) else str(query_bytes)
            result = firewall.check_request(
                client_ip=client_ip,
                method=scope.get("method", "GET"),
                path=scope.get("path", "/"),
                headers=headers,
                body=body,
                query=query,
                ssl_object=scope.get("ssl_object"),
            )
        except Exception:
            await self._reject(send, 503, "Firewall evaluation failed")
            return

        if not result.allowed:
            await self._reject(send, result.status_code, result.reason or "Request blocked")
            return

        delivered = False

        async def replay_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        try:
            await self.app(scope, replay_receive, send)
        finally:
            security.release_concurrent(client_ip)
