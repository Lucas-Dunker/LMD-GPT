"""
Merge LoRA adapter into the base model and prepare it for Ollama.

Pipeline:
  1. Load base model in bf16 on CPU (avoids VRAM pressure during merge)
  2. Load LoRA adapter and merge weights
  3. Save merged HuggingFace model
  4. Print instructions for converting to GGUF and loading into Ollama
"""
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL, LORA_DIR, MERGED_DIR


def merge_and_save(
    lora_dir: Path = LORA_DIR,
    output_dir: Path = MERGED_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {BASE_MODEL}  (on CPU, may take a minute)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(model, str(lora_dir))

    print("Merging weights…")
    model = model.merge_and_unload()

    print(f"Saving merged model → {output_dir}")
    model.save_pretrained(str(output_dir))

    tokenizer = AutoTokenizer.from_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(output_dir))

    _print_next_steps(output_dir)
    return output_dir


def _print_next_steps(merged_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("Merged model saved. To load it into Ollama:")
    print()
    print("  # 1. Clone and build llama.cpp (if not already done)")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp")
    print("  pip install -r requirements.txt")
    print("  cmake -B build && cmake --build build --config Release -j $(nproc)")
    print("  # llama-quantize binary will be at: build/bin/llama-quantize")
    print()
    print("  # 2a. Fix tokenizer config (if you see a TokenizersBackend error)")
    print(f"  python3 -c \"")
    print(f"  import json")
    print(f"  path = '{merged_dir}/tokenizer_config.json'")
    print(f"  d = json.load(open(path))")
    print(f"  d['tokenizer_class'] = 'PreTrainedTokenizerFast'")
    print(f"  [d.pop(k, None) for k in ['backend', 'is_local', 'max_length', 'stride', 'truncation_side', 'truncation_strategy']]")
    print(f"  json.dump(d, open(path, 'w'), indent=2)")
    print(f"  \"")
    print()
    print("  # 2b. Convert to F16 GGUF (quantization is a separate step)")
    print(f"  python convert_hf_to_gguf.py {merged_dir} \\")
    print(f"      --outfile {merged_dir}/lmd-gpt-f16.gguf \\")
    print("      --outtype f16")
    print()
    print("  # 2c. Quantize to Q4_K_M")
    print(f"  build/bin/llama-quantize \\")
    print(f"      {merged_dir}/lmd-gpt-f16.gguf \\")
    print(f"      {merged_dir}/lmd-gpt.gguf \\")
    print("      Q4_K_M")
    print()
    print("  # 3. Create an Ollama Modelfile")
    print(f'  echo \'FROM {merged_dir}/lmd-gpt.gguf\' > Modelfile')
    print()
    print("  # 4. Register with Ollama")
    print("  ollama create lmd-gpt -f Modelfile")
    print()
    print("  # 5. Update config.py:")
    print('  INFERENCE_MODEL = "lmd-gpt"')
    print("=" * 60)
