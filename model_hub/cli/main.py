#!/usr/bin/env python3
"""
SNEPPX Model Hub — CLI Tool

Usage:
    sneppx-hub list [OPTIONS]
    sneppx-hub search QUERY [OPTIONS]
    sneppx-hub info MODEL_NAME [OPTIONS]
    sneppx-hub upload MODEL_DIR [OPTIONS]
    sneppx-hub download MODEL_NAME [OPTIONS]
    sneppx-hub leaderboard [OPTIONS]
    sneppx-hub orgs list
    sneppx-hub login USER PASSWORD
    sneppx-hub keys create USERNAME [OPTIONS]

Environment:
    SNEPPX_HUB_URL     Hub API URL (default: http://localhost:8000)
    SNEPPX_HUB_API_KEY API key for authentication
    SNEPPX_HUB_CACHE   Local cache directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add the model_hub package to the path if running standalone
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_api_url() -> str:
    return os.environ.get("SNEPPX_HUB_URL", "http://localhost:8000")


def _get_api_key() -> Optional[str]:
    key = os.environ.get("SNEPPX_HUB_API_KEY")
    if key:
        return key
    cache_dir = os.environ.get("SNEPPX_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sneppx", "hub"))
    token_path = os.path.join(cache_dir, "token")
    if os.path.exists(token_path):
        with open(token_path) as f:
            return f.read().strip()
    return None


def _ensure_client():
    from ..api.client import HubAPIClient
    url = _get_api_url()
    key = _get_api_key()
    return HubAPIClient(base_url=url, api_key=key)


def _format_model_card(card: Dict[str, Any], verbose: bool = False) -> str:
    lines = []
    name = card.get("name", "unknown")
    version = card.get("version", "")
    org = card.get("organization", "")

    title = f"{name}@{version}" if version else name
    if org and org not in name:
        title = f"{org}/{title}"

    lines.append(f"\n{'='*60}")
    lines.append(f"  {title}")
    lines.append(f"{'='*60}")

    desc = card.get("description", "")
    if desc:
        lines.append(f"  Description: {desc}")

    lines.append(f"  Architecture: {card.get('architecture', 'N/A')}")
    lines.append(f"  Format: {card.get('format', 'N/A')}")
    lines.append(f"  Task: {card.get('task', 'N/A')}")
    lines.append(f"  License: {card.get('license', 'N/A')}")
    lines.append(f"  Visibility: {card.get('visibility', 'public')}")

    tags = card.get("tags", [])
    if tags:
        lines.append(f"  Tags: {', '.join(tags)}")

    lang = card.get("language")
    if lang:
        lines.append(f"  Language: {lang}")

    datasets = card.get("datasets", [])
    if datasets:
        lines.append(f"  Datasets: {', '.join(datasets)}")

    author = card.get("author")
    if author:
        lines.append(f"  Author: {author}")

    created = card.get("created_at", "")
    if created:
        lines.append(f"  Created: {created}")

    total_size = card.get("total_size", 0)
    from model_hub.utils import format_file_size
    lines.append(f"  Size: {format_file_size(total_size)}")

    if verbose:
        files = card.get("files", [])
        if files:
            lines.append(f"\n  Files:")
            for f in files:
                lines.append(f"    - {f['filename']} ({format_file_size(f.get('size', 0))})")

        reqs = card.get("requirements", {})
        if reqs:
            lines.append(f"\n  Requirements:")
            if reqs.get("min_ram"):
                lines.append(f"    RAM: {format_file_size(reqs['min_ram'])}")
            if reqs.get("min_disk"):
                lines.append(f"    Disk: {format_file_size(reqs['min_disk'])}")
            if reqs.get("gpu_memory"):
                lines.append(f"    GPU: {format_file_size(reqs['gpu_memory'])}")
            if reqs.get("python_version"):
                lines.append(f"    Python: {reqs['python_version']}")

        training = card.get("training_config")
        if training:
            lines.append(f"\n  Training Config:")
            for k, v in training.items():
                if v is not None:
                    lines.append(f"    {k}: {v}")

    lines.append("")
    return "\n".join(lines)


def _format_model_list(models: List[Dict], show_size: bool = False) -> str:
    from model_hub.utils import format_file_size

    if not models:
        return "No models found."

    lines = []
    for m in models:
        name = m.get("name", "unknown")
        version = m.get("version", "")
        size = m.get("total_size", m.get("_size", 0))
        size_str = format_file_size(size) if show_size else ""

        task = m.get("task", "unknown")
        tags = m.get("tags", [])
        tag_str = f" [{', '.join(tags[:3])}]" if tags else ""

        line = f"  {name}@{version} ({task})"
        if size_str:
            line += f" [{size_str}]"
        line += tag_str
        lines.append(line)

    return "\n".join(lines)


# ---- Commands ----

def cmd_list(args: argparse.Namespace):
    client = _ensure_client()
    params = {}
    if args.query:
        params["q"] = args.query
    if args.task:
        params["task"] = args.task
    if args.tag:
        params["tag"] = args.tag
    if args.org:
        params["org"] = args.org

    result = client._request("GET", "/api/v1/models", params=params)
    models = result.get("items", result)
    print(_format_model_list(models, show_size=args.size))
    print(f"\n  Total: {result.get('total', len(models))} models")


def cmd_search(args: argparse.Namespace):
    client = _ensure_client()
    params = {"q": args.query, "limit": args.limit}
    if args.task:
        params["task"] = args.task
    if args.tags:
        params["tags"] = args.tags

    result = client._request("GET", "/api/v1/search", params=params)
    results = result.get("results", result)
    print(_format_model_list(results, show_size=args.size))


def cmd_info(args: argparse.Namespace):
    client = _ensure_client()
    try:
        card = client.get_model(args.model_name, args.version)
    except Exception:
        card = None
    if card:
        print(_format_model_card(card.model_dump(mode="json"), verbose=args.verbose))
    else:
        print(f"Model not found: {args.model_name}@{args.version}")
        sys.exit(1)


def cmd_versions(args: argparse.Namespace):
    client = _ensure_client()
    versions = client.get_model_versions(args.model_name)
    if versions:
        print(f"Versions of {args.model_name}:")
        for v in versions:
            print(f"  {v.version}")
    else:
        print(f"No versions found for: {args.model_name}")


def cmd_upload(args: argparse.Namespace):
    client = _ensure_client()
    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Error: Directory not found: {model_dir}")
        sys.exit(1)

    # Read model card
    card_path = os.path.join(model_dir, "model_card.json")
    if not os.path.exists(card_path):
        print("Error: model_card.json not found in the model directory")
        print("  Create one with: sneppx-hub init <model_name>")
        sys.exit(1)

    with open(card_path) as f:
        card_data = json.load(f)

    files = {}
    for filename in os.listdir(model_dir):
        if filename == "model_card.json" or filename.endswith(".py"):
            continue
        filepath = os.path.join(model_dir, filename)
        if os.path.isfile(filepath):
            files[filename] = filepath

    if not files:
        print("No model files found to upload")
        sys.exit(1)

    # Upload metadata first
    org = card_data.get("organization")
    card_name = card_data.get("name", args.model_dir)
    model_name = f"{org}/{card_name}" if org and org not in card_name else card_name
    if args.version:
        card_data["version"] = args.version
    if args.task:
        card_data["task"] = args.task
    if args.format:
        card_data["format"] = args.format
    if args.visibility:
        card_data["visibility"] = args.visibility
    if args.license:
        card_data["license"] = args.license
    version = card_data.get("version")
    resp = client._request(
        "POST", "/api/v1/models/upload",
        data={"card": json.dumps(card_data)},
    )
    print(f"Metadata uploaded: {resp.get('status')}")

    # Upload files
    for filename, filepath in files.items():
        size = os.path.getsize(filepath)
        print(f"  Uploading {filename} ({_format_file_size(size)})...", end="", flush=True)

        with open(filepath, "rb") as f:
            resp = client._request(
                "POST",
                f"/api/v1/models/{client._model_segment(model_name)}/{version}/files/{filename}",
                files={"file": (filename, f, "application/octet-stream")},
            )
        print(f" OK ({resp.get('sha256', '')[:12]}...)")

    print(f"\nUpload complete: {len(files)} files")
    print(f"  Model: {model_name}")
    print(f"  Version: {version}")


def cmd_download(args: argparse.Namespace):
    from model_hub.api.registry import ModelRegistry

    registry = ModelRegistry(client=_ensure_client())
    download_dir = args.output or None

    print(f"Downloading {args.model_name}@{args.version}...")
    try:
        result_dir = registry.download_model(args.model_name, args.version, download_dir)
        print(f"Downloaded to: {result_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_leaderboard(args: argparse.Namespace):
    client = _ensure_client()
    params = {}
    if args.task:
        params["task"] = args.task
    if args.metric:
        params["metric"] = args.metric

    data = client._request("GET", "/api/v1/leaderboard", params=params)
    entries = data.get("entries", [])

    if not entries:
        print("No leaderboard entries.")
        return

    print(f"\n{'Rank':<6} {'Model':<30} {'Score':<12} {'Dataset':<20} {'Task':<25}")
    print("-" * 93)
    for e in sorted(entries, key=lambda x: x["score"], reverse=True):
        rank = e.get("rank", "?")
        name = e.get("model_name", "?")[:29]
        score = f"{e['score']:.4f}"[:11]
        dataset = e.get("dataset", "?")[:19]
        task = e.get("task", "?")[:24]
        print(f"{rank:<6} {name:<30} {score:<12} {dataset:<20} {task:<25}")
    print()


def cmd_submit_benchmark(args: argparse.Namespace):
    client = _ensure_client()

    # Build result
    from model_hub.api.models import BenchmarkResult, ModelTask

    result = BenchmarkResult(
        model_name=args.model,
        version=args.version,
        task=ModelTask(args.task),
        metric=args.metric,
        score=args.score,
        dataset=args.dataset,
    )

    success = client.submit_benchmark(result)
    if success:
        print(f"Benchmark submitted for {args.model}@{args.version}")
    else:
        print("Failed to submit benchmark")
        sys.exit(1)


def cmd_orgs(args: argparse.Namespace):
    client = _ensure_client()
    orgs = client._request("GET", "/api/v1/orgs")
    org_list = orgs.get("orgs", orgs)
    if org_list:
        print(f"\n{'Name':<25} {'Description':<45} {'Public':<8}")
        print("-" * 78)
        for org in org_list:
            name = org.get("name", "?")[:24]
            desc = org.get("description", "")[:44]
            public = "Yes" if org.get("is_public", True) else "No"
            print(f"{name:<25} {desc:<45} {public:<8}")
    else:
        print("No organizations found.")


def cmd_login(args: argparse.Namespace):
    client = _ensure_client()
    try:
        result = client._request("POST", "/api/v1/auth/login", data={
            "username": args.username,
            "password": args.password,
        })
    except Exception as e:
        print(f"Login failed: {e}")
        sys.exit(1)
    token = result.get("access_token")
    if token:
        cache_dir = os.environ.get("SNEPPX_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sneppx", "hub"))
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "token"), "w") as f:
            f.write(token)
        print(f"Logged in as {args.username}")
    else:
        print("Login failed")
        sys.exit(1)


def cmd_keys(args: argparse.Namespace):
    client = _ensure_client()
    if args.create:
        result = client._request("POST", "/api/v1/keys", data={
            "username": args.create,
            "scopes": args.scopes or "read,write,models",
        })
        print(f"API key created:")
        print(f"  Key ID: {result.get('key_id')}")
        print(f"  key_id:prefix: {result.get('api_key')}")
        print(f"  Scopes: {result.get('scopes')}")
    else:
        result = client._request("GET", "/api/v1/keys/me")
        keys = result.get("keys", [])
        if keys:
            print(f"\n{'Key ID':<20} {'Created':<26} {'Last Used':<26}")
            print("-" * 72)
            for k in keys:
                kid = k.get("key_id", "?")[:19]
                created = k.get("created_at", "?")[:25]
                used = k.get("last_used", "never")[:25] if k.get("last_used") else "never"
                print(f"{kid:<20} {created:<26} {used:<26}")
        else:
            print("No API keys found.")


def cmd_init(args: argparse.Namespace):
    """Initialize a model directory structure with template files."""
    from model_hub.api.models import ModelCard, ModelTask, ModelFormat, ModelVisibility, License

    model_name = args.model_name
    org = model_name.split("/")[0] if "/" in model_name else None
    name = model_name.split("/")[-1] if "/" in model_name else model_name

    template_card = {
        "name": name,
        "organization": org,
        "description": "",
        "version": "v1.0.0",
        "architecture": "",
        "format": ModelFormat.SNEPPX_NATIVE.value,
        "task": args.task or ModelTask.TEXT_GENERATION.value,
        "license": License.MIT.value,
        "tags": [],
        "language": "en",
        "datasets": [],
        "visibility": ModelVisibility.PUBLIC.value,
    }

    os.makedirs(model_name, exist_ok=True)
    card_path = os.path.join(model_name, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(template_card, f, indent=2)

    print(f"Initialized model directory: {model_name}")
    print(f"  Created: {card_path}")
    print(f"\nNext steps:")
    print(f"  1. Edit model_card.json with your model metadata")
    print(f"  2. Copy model weights into {model_name}/")
    print(f"  3. Run: sneppx-hub upload {model_name}")


def cmd_stats(args: argparse.Namespace):
    client = _ensure_client()
    stats = client._request("GET", "/api/v1/stats")
    print(f"\nSNEPPX Model Hub Statistics:")
    print(f"  Total models: {stats.get('total_models', 0)}")
    print(f"  Total versions: {stats.get('total_versions', 0)}")
    print(f"  Total size: {stats.get('total_size_human', '0 B')}")
    print(f"  Total users: {stats.get('total_users', 0)}")
    print(f"  Total orgs: {stats.get('total_orgs', 0)}")
    print(f"  Leaderboard entries: {stats.get('leaderboard_entries', 0)}")
    print()


def _format_file_size(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        prog="sneppx-hub",
        description="SNEPPX Model Hub CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # list
    p_list = sub.add_parser("list", help="List available models")
    p_list.add_argument("--query", "-q", type=str, default=None)
    p_list.add_argument("--task", "-t", type=str, default=None)
    p_list.add_argument("--tag", type=str, default=None)
    p_list.add_argument("--org", type=str, default=None)
    p_list.add_argument("--size", action="store_true")
    p_list.set_defaults(func=cmd_list)

    # search
    p_search = sub.add_parser("search", help="Search for models")
    p_search.add_argument("query", type=str)
    p_search.add_argument("--task", "-t", type=str, default=None)
    p_search.add_argument("--tags", type=str, default=None)
    p_search.add_argument("--limit", type=int, default=50)
    p_search.add_argument("--size", action="store_true")
    p_search.set_defaults(func=cmd_search)

    # info
    p_info = sub.add_parser("info", help="Show detailed info about a model")
    p_info.add_argument("model_name", type=str)
    p_info.add_argument("--version", default="latest")
    p_info.add_argument("--verbose", "-v", action="store_true")
    p_info.set_defaults(func=cmd_info)

    # versions
    p_ver = sub.add_parser("versions", help="List available versions of a model")
    p_ver.add_argument("model_name", type=str)
    p_ver.set_defaults(func=cmd_versions)

    # upload
    p_up = sub.add_parser("upload", help="Upload a model")
    p_up.add_argument("model_dir", type=str)
    p_up.add_argument("--version", default="v1.0.0")
    p_up.add_argument("--task", "-t", type=str, default=None)
    p_up.add_argument("--format", default="sneppx-native")
    p_up.add_argument("--visibility", default="public")
    p_up.add_argument("--license", default="mit")
    p_up.set_defaults(func=cmd_upload)

    # download
    p_dl = sub.add_parser("download", help="Download a model")
    p_dl.add_argument("model_name", type=str)
    p_dl.add_argument("--version", default="latest")
    p_dl.add_argument("--output", "-o", type=str, default=None)
    p_dl.set_defaults(func=cmd_download)

    # leaderboard
    p_lb = sub.add_parser("leaderboard", help="View or submit to the leaderboard")
    p_lb.add_argument("--task", type=str, default=None)
    p_lb.add_argument("--metric", type=str, default=None)
    p_lb.set_defaults(func=cmd_leaderboard)

    # submit-benchmark
    p_sub = sub.add_parser("submit-benchmark", help="Submit a benchmark result")
    p_sub.add_argument("--model", required=True)
    p_sub.add_argument("--version", required=True)
    p_sub.add_argument("--task", required=True)
    p_sub.add_argument("--metric", required=True)
    p_sub.add_argument("--score", type=float, required=True)
    p_sub.add_argument("--dataset", required=True)
    p_sub.set_defaults(func=cmd_submit_benchmark)

    # orgs
    p_orgs = sub.add_parser("orgs", help="List organizations")
    p_orgs.set_defaults(func=lambda args: cmd_orgs(args))
    # Subcommand for listing orgs
    p_orgs_sub = sub.add_parser("orgs-list", help="List organizations")
    p_orgs_sub.set_defaults(func=cmd_orgs)

    # login
    p_login = sub.add_parser("login", help="Login to the hub")
    p_login.add_argument("username", type=str)
    p_login.add_argument("password", type=str)
    p_login.set_defaults(func=cmd_login)

    # keys
    p_keys = sub.add_parser("keys", help="Manage API keys")
    p_keys.add_argument("--create", type=str, default=None, help="Create a new key for username")
    p_keys.add_argument("--scopes", type=str, default="read,write,models")
    p_keys.set_defaults(func=cmd_keys)

    # init
    p_init = sub.add_parser("init", help="Initialize a model directory")
    p_init.add_argument("model_name", type=str)
    p_init.add_argument("--task", type=str, default=None)
    p_init.set_defaults(func=cmd_init)

    # stats
    p_stats = sub.add_parser("stats", help="Show hub statistics")
    p_stats.set_defaults(func=cmd_stats)

    # server (start the API server)
    p_srv = sub.add_parser("server", help="Start the model hub server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--db", default=None)
    p_srv.add_argument("--storage-backend", default="local")
    p_srv.add_argument("--storage-path", default=None)
    p_srv.add_argument("--workers", type=int, default=1)
    p_srv.set_defaults(func=cmd_server)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


def cmd_server(args: argparse.Namespace):
    """Start the model hub API server."""
    import uvicorn
    from ..api.main import create_app, HubDatabase, StorageConfig, get_storage_backend

    db_path = args.db or os.path.join(
        os.environ.get("SNEPPX_HUB_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "sneppx", "hub")),
        "data"
    )

    storage_config = StorageConfig(
        backend=args.storage_backend,
        base_path=args.storage_path or os.path.join(db_path, "storage"),
    )

    app = create_app(db_path, storage_config)

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    main()
