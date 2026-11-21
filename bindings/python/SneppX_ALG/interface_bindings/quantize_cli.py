"""CLI launcher for model quantization (FP8/INT4/AWQ/GPTQ)."""

import argparse
import json
import logging


def main():
    parser = argparse.ArgumentParser(description="SneppX Model Quantization")
    parser.add_argument("--mode", type=str, default="int4", choices=["int4", "int8", "fp8", "awq", "gptq"],
                        help="Quantization mode")
    parser.add_argument("--granularity", type=str, default="per_channel", choices=["per_tensor", "per_channel"],
                        help="Quantization granularity")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path for quantized weights")
    parser.add_argument("--skip-layers", type=str, nargs="*", default=["lm_head", "embed_tokens"],
                        help="Layer name substrings to skip")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from .quantization import QuantMode, QuantGranularity
    from .quantized_serve import QuantizedModelConfig, quantize_model_weights, estimate_model_size_mb

    mode_map = {
        "int4": QuantMode.INT4_SYM,
        "int8": QuantMode.INT8_SYM,
        "fp8": QuantMode.FP8_E4M3,
        "awq": QuantMode.AWQ,
        "gptq": QuantMode.GPTQ,
    }
    gran_map = {
        "per_tensor": QuantGranularity.PER_TENSOR,
        "per_channel": QuantGranularity.PER_CHANNEL,
    }

    cfg = QuantizedModelConfig(
        quant_mode=mode_map[args.mode],
        quant_granularity=gran_map[args.granularity],
        skip_layers=args.skip_layers,
    )

    with open(args.model_config) as f:
        config = json.load(f)

    model_params = {}
    for name, w in config.get("weights", {}).items():
        import numpy as np
        model_params[name] = np.array(w, dtype=np.float32)

    logging.info(f"Quantizing {len(model_params)} layers with mode={args.mode}...")
    quantized = quantize_model_weights(model_params, cfg)
    size_mb = estimate_model_size_mb(quantized)
    logging.info(f"Quantized model size: {size_mb:.2f} MB")

    if args.output:
        import pickle
        with open(args.output, "wb") as f:
            pickle.dump(quantized, f)
        logging.info(f"Saved quantized weights to {args.output}")


if __name__ == "__main__":
    main()
