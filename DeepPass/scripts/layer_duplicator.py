"""
DeepPass Layer Duplicator

Implements the RYS (Repeat Your Self) layer duplication method from David Noel Ng's work.
Given a model and a configuration (i, j), duplicates layers i through j in the forward pass
without modifying any weights.

For a model with N layers and config (i, j):
  Normal:     0 → 1 → ... → j-1 → j → ... → N-1
  Duplicated: 0 → ... → j-1 → [i → ... → j-1] → j → ... → N-1

  Layers [i, j) execute twice. The second pass reuses the same weights (no extra VRAM).

Example: (i=2, j=7) for N=9:
  0 → 1 → 2 → 3 → 4 → 5 → 6 → [2 → 3 → 4 → 5 → 6] → 7 → 8
  duplicated: [2, 3, 4, 5, 6]

Approach: Instead of modifying the ModuleList (which breaks KV cache indexing),
we monkey-patch the inner model's forward to manually iterate through the
duplicated layer sequence with use_cache=False to avoid cache issues.
"""

import torch
import functools
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_layer_duplication(model, i: int, j: int):
    """
    Apply layer duplication to a HuggingFace model by monkey-patching the
    inner model's forward pass.

    This avoids modifying the ModuleList, so KV cache and position IDs
    stay consistent. We instead override the forward to manually run
    the layer sequence: [0..j-1, i..j-1, j..N-1].

    Args:
        model: HuggingFace AutoModelForCausalLM
        i: start of duplicated block (inclusive)
        j: end of duplicated block (exclusive)
    """
    inner_model = model.model if hasattr(model, 'model') else model.transformer
    layers = inner_model.layers if hasattr(inner_model, 'layers') else inner_model.h
    N = len(layers)

    assert 0 <= i < j <= N, f"Invalid config: i={i}, j={j}, N={N}"

    # Build the execution order
    layer_order = list(range(j)) + list(range(i, j)) + list(range(j, N))

    # Store original forward
    original_forward = inner_model.forward

    @functools.wraps(original_forward)
    def patched_forward(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        # Force disable cache for duplicated models to avoid index mismatches
        return original_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,  # disable cache
            inputs_embeds=inputs_embeds,
            use_cache=False,  # disable cache
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

    # Actually, the cleaner approach: replace the layer list with the
    # duplicated sequence, but also disable caching entirely so there's
    # no cache slot mismatch.
    import torch.nn as nn
    original_layers = list(layers)
    new_layer_sequence = [layers[idx] for idx in layer_order]

    attr_name = 'layers' if hasattr(inner_model, 'layers') else 'h'
    setattr(inner_model, attr_name, nn.ModuleList(new_layer_sequence))

    new_num_layers = len(new_layer_sequence)
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = new_num_layers

    return N, new_num_layers, j - i


def restore_layers(model, original_layers, original_num_layers):
    """Restore original layer configuration."""
    import torch.nn as nn
    inner_model = model.model if hasattr(model, 'model') else model.transformer
    attr_name = 'layers' if hasattr(inner_model, 'layers') else 'h'
    setattr(inner_model, attr_name, nn.ModuleList(original_layers))
    if hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = original_num_layers


def generate_no_cache(model, tokenizer, prompt, max_new_tokens=64):
    """
    Generate text token-by-token without KV cache.
    Slower than cached generation but works correctly with duplicated layers.
    """
    # For multi-GPU models, model.device may not work; use first parameter's device
    _device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(_device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False)

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Decode only generated tokens (skip the prompt)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    generated = input_ids[0, prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def load_original_model(
    model_path: str,
    device_map: str = "auto",
    dtype=torch.bfloat16,
    trust_remote_code: bool = True,
):
    """Load original model without duplication (for baseline comparison)."""
    print(f"Loading baseline model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    inner = model.model if hasattr(model, 'model') else model.transformer
    # Gemma3 nests the text model: model.model.language_model.layers
    if hasattr(inner, 'language_model'):
        inner = inner.language_model
    layers = inner.layers if hasattr(inner, 'layers') else inner.h
    num_layers = len(layers)
    print(f"Loaded: {num_layers} layers")
    return model, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test layer duplication on a model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--i", type=int, required=True, help="Start of duplicated block")
    parser.add_argument("--j", type=int, required=True, help="End of duplicated block")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?",
                        help="Test prompt")
    args = parser.parse_args()

    model, tokenizer = load_original_model(args.model)
    apply_layer_duplication(model, args.i, args.j)
    response = generate_no_cache(model, tokenizer, args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")
