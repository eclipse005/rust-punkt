#!/usr/bin/env python3
"""Regenerate rust-punkt's per-language training data from current NLTK punkt_tab.

Usage:
    python scripts/refresh_punkt_data.py --dry-run   # show diff only, do not write
    python scripts/refresh_punkt_data.py             # write all languages
    python scripts/refresh_punkt_data.py --lang english  # one language at a time

Output: one row per language with old/new size and key field counts.
"""
import argparse
import json
import sys
from pathlib import Path

# IMPORTANT: keep in sync with the preloaded_data!() invocations in
# src/trainer.rs. Adding a new language requires updates in both places.
LANGS = [
    "czech", "danish", "dutch", "english", "estonian", "finnish",
    "french", "german", "greek", "italian", "norwegian", "polish",
    "portuguese", "slovene", "spanish", "swedish", "turkish",
]


def fetch_params(lang: str):
    """Fetch NLTK's current punkt_tab parameters for the given language."""
    import nltk
    from nltk.tokenize.punkt import PunktTokenizer

    # NLTK 3.8.2+ ships data as `punkt_tab`; older versions use `punkt`.
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
            break
        except Exception:
            continue
    return PunktTokenizer(lang)._params


def to_json(params) -> dict:
    """Convert an NLTK PunktParameters into the rust-punkt JSON schema."""
    ortho = {}
    for k, flags in params.ortho_context.items():
        # NLTK 3.8.x: value is a set of IntFlag (OR them together).
        # NLTK 3.9.x: value is a single IntFlag / int (already combined).
        if isinstance(flags, (set, frozenset, list, tuple)):
            v = 0
            for f in flags:
                v |= int(f.value if hasattr(f, "value") else f)
        else:
            v = int(flags.value if hasattr(flags, "value") else flags)
        if v:
            ortho[k] = v
    return {
        "sentence_starters": sorted(params.sent_starters),
        "abbrev_types": sorted(params.abbrev_types),
        "collocations": sorted([list(c) for c in params.collocations]),
        "ortho_context": ortho,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print diff stats only, do not write files")
    ap.add_argument("--data-dir", default="src/data",
                    help="rust-punkt data directory (default: src/data)")
    ap.add_argument("--lang", action="append",
                    help="only refresh the specified language (repeatable)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} does not exist", file=sys.stderr)
        return 1

    langs = args.lang if args.lang else LANGS

    header = (
        f"{'lang':<12} {'old KB':>8} {'new KB':>8} "
        f"{'abbrs':>7} {'colloc':>7} {'st-start':>8} {'ortho':>8}"
    )
    print(header)
    print("-" * len(header))

    failures = []
    for lang in langs:
        old_path = data_dir / f"{lang}.json"
        old_size = (old_path.stat().st_size // 1024) if old_path.exists() else 0

        try:
            params = fetch_params(lang)
        except Exception as e:
            print(f"{lang:<12}  ERROR: {e}", file=sys.stderr)
            failures.append((lang, str(e)))
            continue

        new = to_json(params)
        new_str = json.dumps(new, indent=2, ensure_ascii=False)
        new_size = len(new_str) // 1024

        print(
            f"{lang:<12} {old_size:>8} {new_size:>8} "
            f"{len(new['abbrev_types']):>7} {len(new['collocations']):>7} "
            f"{len(new['sentence_starters']):>8} {len(new['ortho_context']):>8}"
        )

        if not args.dry_run:
            old_path.write_text(new_str + "\n", encoding="utf-8")

    print()
    if args.dry_run:
        print("(dry-run, no files written)")
    else:
        print(f"updated {len(langs) - len(failures)}/{len(langs)} files in {data_dir}/")
    if failures:
        print(f"failures: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())