# SneppX Console

`sneppx-console` is the interactive, tab-autocompleted REPL for exploring the
SNEPPX-Algo Python API without writing a script. It is built on the `cmd2`
library (via `SneppX_ALG.interface_bindings.vizmon`), provides history,
syntax-highlighted output, and context-aware help for every command.

## Starting the console

```powershell
$env:PYTHONPATH = "bindings/python"
python -m SneppX_ALG.interface_bindings.sneppx_console
```

If the package is installed (`pip install -e .`), the entry point is on PATH:

```bash
sneppx-console
```

You should see:

```
SNEPPX-Algo Console  v1.1.1   (C backend: True)
Type help or ? for command list.
sneppx> █
```

The banner reports whether the C backend (`_HAS_C_BACKEND`) is loaded. If
`False`, tensor ops still work via the NumPy fallback, but algorithm-stage
methods (`HSSModel.forward`, `Trainer.train_step`, etc.) will raise
`RuntimeError: C backend not available`.

## Tab completion

Tab completion is available for **commands**, **arguments**, and **symbols**.

```
sneppx> from SneppX_ALG import Trans <TAB>
sneppx> # completes → Transformer  TransformerBlock
sneppx> load_model --name <TAB>
sneppx> # completes → llama-2-7b mistral-7b qwen2-7b deepseek-v2-lite
```

| Trigger | Completes |
|---------|-----------|
| `Ctrl-Space` or `TAB` at start of line | command name |
| `TAB` after `from SneppX_ALG import ` | exported symbol names |
| `TAB` after `--name ` on `load_model` | known model IDs |
| `TAB` after a function call `...` | nothing (use `dir()` instead) |

## Commands

| Command | Description |
|---------|-------------|
| `Tensor.zeros SHAPE` | Create a zero tensor. `Tensor.zeros 4 8` → shape `(4,8)`. |
| `Tensor.randn SHAPE [dtype] [cuda]` | Random tensor. |
| `run FILE [--backend cuda|cpu]` | Execute a SneppX script / notebook cell file. |
| `load_model --name llama-2-7b [--cache-dir DIR]` | Fetch + load a model config (no weights without C backend). |
| `serve --port 8000 [--host 127.0.0.1] [--auth-key K]` | Start the FastAPI inference server in-process. |
| `scan PATH --format c|hpp` | Run `sneppx-analyze` security scan on a source path. |
| `profile --duration 10 --output report.json` | Record a 10-s profile to JSON. |
| `checkpoint save PATH` / `checkpoint load PATH` | Persist / restore trainer state. |
| `distributed --world 4 --backend nccl` | Print the launch env for a 4-rank job. |
| `quantize --mode int4|int8|fp8 --model NAME` | Quantize a loaded model in place. |
| `list_backends` | Show detected CUDA/NCCL/CPU feature flags. |
| `exit` / `quit` | Leave the console. |

## Built-in help

```
sneppx> help
sneppx> serve --help
sneppx> scan --help
sneppx> Tensor.zeros --help
```

Help text mirrors the docstring of the underlying binding, so it stays in
sync with the code.

## Example session

```
sneppx> Tensor.zeros 4 8
Tensor(shape=(4, 8), dtype=float32, device=cpu)

sneppx> from SneppX_ALG import Tensor, Linear, AdamW
sneppx> x = Tensor.randn 4 8
sneppx> lin = Linear(8, 16)
sneppx> y = lin.forward(x)
sneppx> y.shape
(4, 16)

sneppx> list_models
Available: llama-2-7b, llama-3-8b, mistral-7b, qwen2-7b, deepseek-v2-lite

sneppx> load_model --name mistral-7b
[SNEPPX from_pretrained] mistral-7b -> family=mistral, size=7B
  hidden_size=4096, layers=32, heads=32, kv_heads=8
config loaded (no weights — build C backend for inference)

sneppx> profile --duration 5 --output /tmp/prof.json
profiling 5.0s ... done
wrote /tmp/prof.json (32 entries)

sneppx> quantize --mode int4 --model mistral-7b
[quantize] INT4 sym  | weight=7.21 GB -> 3.61 GB (quantize_error snr=32.4 dB)

sneppx> exit
bye.
```

## Scripting the console

Every command is a `do_<name>` method on the
`SneppX_ALG.interface_bindings.sneppx_console.SneppXConsole` class, so you can
subclass and override behavior — useful for CI smoke tests or custom
workflows. Pass `--file script.sneppx` to run a batch of commands non-
interactively (no TTY required).

## Limitations

- The console is a thin wrapper; heavy compute still runs through the C
  backend. For headless use, prefer the CLI tools (`sneppx-train`,
  `sneppx-serve`, `sneppx-quantize`, `sneppx-analyze`) directly.
- CUDA tensors require a real GPU + NCCL; the console reports
  `cuda_is_available()` on startup so you can branch in scripts.
