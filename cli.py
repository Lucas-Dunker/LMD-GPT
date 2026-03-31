"""
LMD-GPT CLI — Personal AI trained on your data.

Commands:
  ingest   [--sources obsidian discord gdocs gcal]  Embed data into vector store
  chat     [--source <filter>]                       RAG chat loop
  finetune prepare  [--username NAME]                Build training dataset
  finetune train    [--resume CHECKPOINT]            QLoRA fine-tune
  finetune export                                    Merge + prep for Ollama
  status                                             Show system + store info
"""
import argparse
import sys


# -----------------------------------------------------------------------
# Command handlers
# -----------------------------------------------------------------------

def cmd_ingest(args):
    from embeddings.store import VectorStore
    store = VectorStore()

    sources = args.sources or ["obsidian", "discord", "gdocs", "gcal"]

    if "obsidian" in sources:
        from ingestion.obsidian import load_all
        print("Ingesting Obsidian notes…")
        docs = load_all()
        n = store.add_documents(docs, chunk=True, source="obsidian")
        print(f"  → {n} chunks from {len(docs)} notes")

    if "discord" in sources:
        from ingestion.discord import load_all
        print("Ingesting Discord messages…")
        docs = load_all()
        # Messages are already short — don't chunk
        n = store.add_documents(docs, chunk=False, source="discord")
        print(f"  → {n} messages")

    if "gdocs" in sources:
        from ingestion.gdocs import load_all
        print("Ingesting Google Docs…")
        docs = load_all()
        n = store.add_documents(docs, chunk=True, source="gdocs")
        print(f"  → {n} chunks from {len(docs)} docs")

    if "gcal" in sources:
        from ingestion.gcal import load_all
        print("Ingesting Google Calendar…")
        docs = load_all()
        n = store.add_documents(docs, chunk=False, source="gcal")
        print(f"  → {n} events")

    print(f"\nVector store total: {store.count():,} chunks")


def cmd_chat(args):
    from rag.chain import stream_chat

    source_label = f" [{args.source}]" if args.source else ""
    print(f"LMD-GPT (RAG mode{source_label}) — type 'exit' to quit\n")

    history: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        print("AI: ", end="", flush=True)
        response_parts: list[str] = []
        try:
            for token in stream_chat(query, source_filter=args.source, history=history):
                print(token, end="", flush=True)
                response_parts.append(token)
        except KeyboardInterrupt:
            pass
        print()

        full_response = "".join(response_parts)
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": full_response})
        history = history[-12:]  # keep last 6 turns


def cmd_finetune_prepare(args):
    from finetune.prepare_data import prepare
    prepare(username=args.username)


def cmd_finetune_train(args):
    from finetune.train import train
    train(resume_from=args.resume)


def cmd_finetune_export(_args):
    from finetune.export import merge_and_save
    merge_and_save()


def cmd_status(_args):
    from embeddings.store import VectorStore
    print("=== Vector Store ===")
    store = VectorStore()
    print(f"  Chunks stored: {store.count():,}")

    print("\n=== GPU / Deps ===")
    # Re-use the check script logic inline
    import scripts.check_gpu as cg
    cg.check_nvidia_smi()
    cg.check_torch_cuda()
    cg.check_bitsandbytes()

    print("\n=== Ollama ===")
    cg.check_ollama()
    cg.check_embedding_model()
    cg.check_inference_model()


# -----------------------------------------------------------------------
# CLI wiring
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="lmd-gpt",
        description="Personal AI built from your own data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Embed data into the vector store")
    p_ingest.add_argument(
        "--sources", nargs="+",
        choices=["obsidian", "discord", "gdocs", "gcal"],
        help="Which sources to ingest (default: all)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Start a RAG chat session")
    p_chat.add_argument(
        "--source",
        choices=["obsidian", "discord", "gdocs", "gcal"],
        help="Restrict retrieval to a single source",
    )
    p_chat.set_defaults(func=cmd_chat)

    # --- finetune ---
    p_ft = sub.add_parser("finetune", help="Fine-tune the model on your data")
    ft_sub = p_ft.add_subparsers(dest="ft_command", required=True)

    p_prep = ft_sub.add_parser("prepare", help="Build the training JSONL")
    p_prep.add_argument("--username", help="Override the name used in training prompts")
    p_prep.set_defaults(func=cmd_finetune_prepare)

    p_train = ft_sub.add_parser("train", help="Run QLoRA training")
    p_train.add_argument("--resume", metavar="CHECKPOINT", help="Resume from a checkpoint dir")
    p_train.set_defaults(func=cmd_finetune_train)

    p_export = ft_sub.add_parser("export", help="Merge LoRA weights and prep for Ollama")
    p_export.set_defaults(func=cmd_finetune_export)

    # --- status ---
    p_status = sub.add_parser("status", help="Show system + vector store status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
