"""CLI launcher for the SneppX inference server."""

import os
import sys
import json
import argparse
import logging
import ipaddress



def _is_loopback_host(host):
    """Return whether a bind host is explicitly loopback-only."""
    normalized = host.strip().strip("[]").lower().split("%", 1)[0]
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # Hostnames other than localhost may resolve to public interfaces.
        return False


def _parse_api_keys(api_keys):
    """Parse comma-separated API keys, discarding empty entries."""
    return [key.strip() for key in (api_keys or "").split(",") if key.strip()]


def _validate_auth_settings(host, auth_mode, api_keys, key_store_path):
    """Reject unauthenticated remote binds and empty enabled auth modes."""
    if auth_mode == "none" and not _is_loopback_host(host):
        raise ValueError(
            "Authentication cannot be disabled on a non-loopback bind address. "
            "Use --auth-mode bearer or api_key with --api-keys or --key-store-path, "
            "or bind to 127.0.0.1/::1 for local-only access."
        )
    configured_keys = _parse_api_keys(api_keys)
    has_environment_keys = (
        auth_mode == "api_key" and bool(os.environ.get("SNEPPX_API_KEYS", "").strip())
    )
    if auth_mode in {"bearer", "api_key"} and not (
        configured_keys or key_store_path or has_environment_keys
    ):
        raise ValueError(
            f"--auth-mode {auth_mode} requires --api-keys, --key-store-path"
            + (" or SNEPPX_API_KEYS." if auth_mode == "api_key" else ".")
        )


def main():
    parser = argparse.ArgumentParser(description="SneppX Inference Server")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (dev)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config JSON/YAML for pre-loading",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="default",
        help="Model ID when loading from --model-config",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Reserved: checkpoint loading is not implemented; use a model-specific loader",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer.json file or directory",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of uvicorn workers"
    )
    parser.add_argument(
        "--auth-mode",
        type=str,
        default=None,
        choices=[None, "none", "bearer", "api_key"],
        help="Authentication mode",
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default=None,
        help="Comma-separated API keys for auth",
    )
    parser.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=None,
        help="Requests per minute limit",
    )
    parser.add_argument(
        "--no-prompt-filter",
        action="store_true",
        help="Disable prompt injection detection",
    )
    parser.add_argument(
        "--no-output-verifier",
        action="store_true",
        help="Disable output verification",
    )
    parser.add_argument(
        "--firewall-config", type=str, default=None,
        help="Path to firewall YAML config",
    )
    parser.add_argument(
        "--firewall-enabled", type=str, default=None,
        choices=["true", "false"],
        help="Enable/disable firewall (overrides config)",
    )
    parser.add_argument(
        "--tls-enabled", action="store_true",
        help="Enable TLS (requires --tls-certfile and --tls-keyfile)",
    )
    parser.add_argument(
        "--tls-certfile", type=str, default=None,
        help="Path to TLS certificate file",
    )
    parser.add_argument(
        "--tls-keyfile", type=str, default=None,
        help="Path to TLS private key file",
    )
    parser.add_argument(
        "--tls-cafile", type=str, default=None,
        help="Path to CA certificate file for mTLS",
    )
    parser.add_argument(
        "--firewall-allowlist", type=str, default=None,
        help="Comma-separated CIDR allowlist",
    )
    parser.add_argument(
        "--firewall-denylist", type=str, default=None,
        help="Comma-separated CIDR denylist",
    )
    parser.add_argument(
        "--firewall-rate-limit", type=int, default=None,
        help="Max requests per IP per minute",
    )
    parser.add_argument(
        "--firewall-knock-ports", type=str, default=None,
        help="Comma-separated port knock sequence",
    )
    parser.add_argument(
        "--key-store-path", type=str, default=None,
        help="Path to SQLite database for persistent API key storage",
    )
    parser.add_argument(
        "--admin-key", type=str, default=None,
        help="Admin API key for managing keys via /v1/admin/keys endpoints",
    )
    parser.add_argument(
        "--usage-db-path", type=str, default=None,
        help="Path to SQLite database for usage tracking",
    )
    args = parser.parse_args()

    if args.checkpoint:
        parser.error(
            "--checkpoint is not implemented by serve_cli; the option is rejected "
            "to prevent serving config-initialized weights by mistake."
        )

    auth_mode = args.auth_mode or "none"
    try:
        _validate_auth_settings(
            args.host, auth_mode, args.api_keys, args.key_store_path
        )
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from .inference_server import set_security
    from .security_middleware import (
        SecurityConfig,
        AuthConfig,
        RateLimitConfig,
        PromptFilterConfig,
        OutputVerifierConfig,
    )

    firewall_overrides = {}
    if args.firewall_config:
        firewall_overrides["config_path"] = args.firewall_config
    if args.firewall_enabled is not None:
        firewall_overrides["enabled"] = args.firewall_enabled == "true"
    if args.tls_enabled:
        firewall_overrides["tls_enabled"] = True
    if args.tls_certfile:
        firewall_overrides["certfile"] = args.tls_certfile
    if args.tls_keyfile:
        firewall_overrides["keyfile"] = args.tls_keyfile
    if args.tls_cafile:
        firewall_overrides["cafile"] = args.tls_cafile
    if args.firewall_allowlist:
        firewall_overrides["allowlist"] = [x.strip() for x in args.firewall_allowlist.split(",")]
    if args.firewall_denylist:
        firewall_overrides["denylist"] = [x.strip() for x in args.firewall_denylist.split(",")]
    if args.firewall_rate_limit:
        firewall_overrides["rate_limit_max"] = args.firewall_rate_limit
    if args.firewall_knock_ports:
        firewall_overrides["knock_ports"] = [int(p.strip()) for p in args.firewall_knock_ports.split(",")]

    auth = AuthConfig(
        mode=auth_mode,
        api_keys=(
            _parse_api_keys(args.api_keys)
        ),
    )
    rate_limit = RateLimitConfig(
        enabled=(args.rate_limit_rpm or 60) > 0,
        requests_per_minute=args.rate_limit_rpm or 60,
    )
    prompt_filter = PromptFilterConfig(enabled=not args.no_prompt_filter)
    output_verifier = OutputVerifierConfig(enabled=not args.no_output_verifier)
    sec_config = SecurityConfig(
        auth=auth,
        rate_limit=rate_limit,
        prompt_filter=prompt_filter,
        output_verifier=output_verifier,
    )
    set_security(
        sec_config,
        firewall_config=firewall_overrides if firewall_overrides else None,
        admin_key=args.admin_key,
        key_store_path=args.key_store_path,
        usage_db_path=args.usage_db_path,
    )
    logging.info(
        f"Security: auth={auth.mode}, rate_limit={rate_limit.requests_per_minute}/min, "
        f"prompt_filter={prompt_filter.enabled}, output_verifier={output_verifier.enabled}"
    )
    if firewall_overrides:
        logging.info(f"Firewall: {firewall_overrides}")

    # Pre-load model if config provided
    if args.model_config:
        _load_model(args)

    _run_server(args)


def _load_model(args):
    from .model_zoo import build_transformer_from_config, get_model_config
    from .inference_server import register_model
    from .tokenizer import Tokenizer
    from .generation import GenerationConfig

    if not os.path.isfile(args.model_config):
        raise FileNotFoundError(f"--model-config not found: {args.model_config}")

    logging.info(f"Loading model config from {args.model_config}")
    with open(args.model_config, encoding="utf-8") as f:
        if args.model_config.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "YAML model config requires PyYAML (`pip install pyyaml`)."
                ) from e
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    if isinstance(config, dict) and "family" in config:
        model_config = get_model_config(config["family"], config.get("size", "7B"))
        model = build_transformer_from_config(model_config)
    elif isinstance(config, dict):
        model = build_transformer_from_config(config)
    else:
        raise ValueError(f"Invalid model config format in {args.model_config}: expected object/dict")

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"--checkpoint not found: {args.checkpoint}")
        raise NotImplementedError(
            "--checkpoint is not supported by serve_cli yet. The model would otherwise "
            "serve config-initialized weights only; load weights with a model-specific "
            "loader before registering the model."
        )

    tokenizer = None
    if args.tokenizer:
        if not os.path.exists(args.tokenizer):
            raise FileNotFoundError(f"--tokenizer not found: {args.tokenizer}")
        tokenizer = Tokenizer(args.tokenizer)

    gen_config = GenerationConfig(
        max_new_tokens=config.get("max_new_tokens", 256) if isinstance(config, dict) else 256,
        temperature=config.get("temperature", 1.0) if isinstance(config, dict) else 1.0,
        top_k=config.get("top_k", 0) if isinstance(config, dict) else 0,
        top_p=config.get("top_p", 1.0) if isinstance(config, dict) else 1.0,
    )

    register_model(
        model_id=args.model_id,
        model=model,
        tokenizer=tokenizer,
        default_config=gen_config,
        meta={"config": config, "checkpoint": args.checkpoint},
    )
    logging.info(f"Model '{args.model_id}' loaded (config-initialized; tokenizer={'yes' if tokenizer else 'no'})")


def _run_server(args):
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    logging.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "SneppX_ALG.interface_bindings.inference_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
