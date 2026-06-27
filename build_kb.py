#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_PATH, KB_JSON_PATH
from rag_fca.knowledge_base import KnowledgeBase


def main():
    parser = argparse.ArgumentParser(
        description="Build the permanent concept-lattice knowledge base.")
    parser.add_argument("--xlsx", default=DATA_PATH,
                        help=f"Source table (default: {os.path.basename(DATA_PATH)})")
    parser.add_argument("--out", default=KB_JSON_PATH,
                        help=f"Output JSON store (default: {KB_JSON_PATH})")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip writing the human-readable FCA review files")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the JSON store already exists")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[build_kb] KB already exists: {args.out}")
        print("[build_kb] Use --force to rebuild. Nothing to do.")
        return

    if not os.path.exists(args.xlsx):
        print(f"[build_kb] ERROR: Source table not found: {args.xlsx}")
        sys.exit(1)

    print("=" * 64)
    print("  CPAM-LLM — Building permanent concept-lattice knowledge base")
    print(f"  Source : {args.xlsx}")
    print(f"  Output : {args.out}")
    print("=" * 64)

    kb = KnowledgeBase()
    kb.build_and_save(
        xlsx_path=args.xlsx,
        json_path=args.out,
        export_review=not args.no_review,
    )

    print("\n[build_kb] OK: Done. The knowledge base is now permanent.")
    print("[build_kb]   Inspect it directly:")
    print(f"[build_kb]     {args.out}")
    if not args.no_review:
        review_dir = os.path.join(os.path.dirname(args.out), "review")
        print(f"[build_kb]   FCA review files: {review_dir}/")
    print("[build_kb]   Next step:  python main.py --show-kb")


if __name__ == "__main__":
    main()
