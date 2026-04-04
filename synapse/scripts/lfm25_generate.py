#!/usr/bin/env python3
"""Generate text with LFM2.5-350M via HuggingFace transformers.

Usage:
    python3 scripts/lfm25_generate.py "What is the capital of France?"
    python3 scripts/lfm25_generate.py  # interactive mode
"""
import sys

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    import torch

    model_id = "LiquidAI/LFM2.5-350M"
    print(f"Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None

    while True:
        if prompt is None:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                continue

        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model.device)

        model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
            streamer=streamer,
        )

        if len(sys.argv) > 1:
            break  # one-shot mode
        prompt = None

if __name__ == "__main__":
    main()
