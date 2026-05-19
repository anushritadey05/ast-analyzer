from __future__ import annotations

import argparse

from .dataset import SyntheticDatasetGenerator, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Big-O dataset")
    parser.add_argument("--count", type=int, default=100, help="Number of samples")
    parser.add_argument("--out", type=str, default="data.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    gen = SyntheticDatasetGenerator()
    samples = gen.generate(args.count)
    write_jsonl(samples, args.out)
    print(f"Wrote {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()
