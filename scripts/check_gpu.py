"""
Verify GPU + dependency setup before running training or inference.
Run with: python scripts/check_gpu.py
"""
import subprocess
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of where this script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _ok(msg): print(f"  \033[32m✓\033[0m {msg}")
def _warn(msg): print(f"  \033[33m!\033[0m {msg}")
def _fail(msg): print(f"  \033[31m✗\033[0m {msg}")


def check_nvidia_smi() -> bool:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                        "--format=csv,noheader"], text=True).strip()
        _ok(f"nvidia-smi: {out}")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        _fail("nvidia-smi not found — check WSL2 GPU passthrough")
        return False


def check_torch_cuda() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            _ok(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | " # type: ignore
                f"{props.name} | {vram_gb:.1f} GB VRAM")
            if vram_gb < 10:
                _warn("Less than 10 GB VRAM — reduce BATCH_SIZE in config.py")
            return True
        else:
            _fail(f"PyTorch {torch.__version__} installed but CUDA not available")
            _warn("Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return False
    except ImportError:
        _fail("PyTorch not installed")
        return False


def check_bitsandbytes() -> bool:
    try:
        import bitsandbytes as bnb
        _ok(f"bitsandbytes {bnb.__version__}")
        return True
    except ImportError:
        _fail("bitsandbytes not installed — needed for QLoRA")
        return False


def check_ollama() -> bool:
    try:
        import ollama
        models = ollama.list()
        names = [m.model for m in models.models]
        _ok(f"Ollama running — {len(names)} model(s): {', '.join(names) or 'none pulled yet'}") # type: ignore
        return True
    except Exception as e:
        _fail(f"Ollama not reachable: {e}")
        _warn("Start Ollama: ollama serve  (in a separate terminal)")
        return False


def check_embedding_model() -> bool:
    try:
        import ollama
        from config import EMBEDDING_MODEL
        resp = ollama.embed(model=EMBEDDING_MODEL, input="test")
        dim = len(resp.embeddings[0])
        _ok(f"Embedding model '{EMBEDDING_MODEL}' working — dim={dim}")
        return True
    except Exception as e:
        _fail(f"Embedding model error: {e}")
        from config import EMBEDDING_MODEL
        _warn(f"Pull it: ollama pull {EMBEDDING_MODEL}")
        return False


def check_inference_model() -> bool:
    try:
        import ollama
        from config import INFERENCE_MODEL
        resp = ollama.chat(
            model=INFERENCE_MODEL,
            messages=[{"role": "user", "content": "Say 'OK' only."}],
        )
        _ok(f"Inference model '{INFERENCE_MODEL}' responding")
        return True
    except Exception as e:
        _fail(f"Inference model error: {e}")
        from config import INFERENCE_MODEL
        _warn(f"Pull it: ollama pull {INFERENCE_MODEL}")
        return False


if __name__ == "__main__":
    print("=== Hardware ===")
    nvidia = check_nvidia_smi()

    print("\n=== PyTorch / CUDA ===")
    torch_ok = check_torch_cuda()

    print("\n=== Fine-tuning deps ===")
    bnb_ok = check_bitsandbytes()

    print("\n=== Ollama ===")
    ollama_ok = check_ollama()

    print("\n=== Models ===")
    if ollama_ok:
        embed_ok = check_embedding_model()
        infer_ok = check_inference_model()
    else:
        _warn("Skipping model checks (Ollama not running)")
        embed_ok = infer_ok = False

    print("\n=== Summary ===")
    rag_ready = ollama_ok and embed_ok and infer_ok
    train_ready = torch_ok and bnb_ok

    if rag_ready and train_ready:
        print("  All systems go — ready for RAG + fine-tuning!")
    else:
        if not rag_ready:
            print("  RAG not ready — fix Ollama / model issues above")
        if not train_ready:
            print("  Fine-tuning not ready — fix PyTorch/bitsandbytes above")
