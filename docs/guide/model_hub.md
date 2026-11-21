# SNEPPX Model Hub Guide

## Overview

The SNEPPX Model Hub (`sneppx-hub`) is a centralized model registry and sharing platform supporting search, versioning, org management, leaderboards, benchmark submissions, and C/Python client integrations.

## Installation

```powershell
pip install -e . --no-deps
```

## CLI Commands

The `sneppx-hub` CLI provides full management and interaction capabilities:

- `sneppx-hub init`: Initialize local cache and configuration.
- `sneppx-hub login`: Authenticate with API key or admin credentials.
- `sneppx-hub upload`: Upload a model with custom version, task, license, and visibility.
- `sneppx-hub download`: Download model weights and configuration.
- `sneppx-hub list`: List registered models with filters (`--tag`, `--org`, `--task`, `--size`).
- `sneppx-hub search`: Search models by query.
- `sneppx-hub info`: View detailed model card and file sizes.
- `sneppx-hub versions`: List all versions of a model.
- `sneppx-hub leaderboard`: View model benchmark scores.
- `sneppx-hub submit-benchmark`: Submit evaluation results.
- `sneppx-hub orgs`: Manage organizations.
- `sneppx-hub keys`: Manage API access keys and scopes.
- `sneppx-hub stats`: View hub statistics.
- `sneppx-hub server`: Launch local registry server.

## Python API

```python
from model_hub import Hub

hub = Hub(url="http://localhost:8100", api_key="...")
hub.upload("org/model", "./weights", version="v1.0.0", task="text-generation")
models = hub.list_models(org="org")
hub.download("org/model@v1.0.0", "./cache")
```

## C Client (`sneppx_hub_client.h`)

Native WinHTTP-based C client for embedded applications and C runtimes:

```c
#include "neural_core/hub_client.h"

SNEPPXHubContext* ctx = SNEPPX_hub_init("http://localhost:8100", "api_key");
SNEPPX_hub_download(ctx, "org/model@v1.0.0", "./local_path");
SNEPPX_hub_free(ctx);
```
