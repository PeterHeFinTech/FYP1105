from __future__ import annotations

import argparse
import csv

from .generator import generate_synthetic_data


def _read_csv(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HMDA synthetic data with CTGAN or TabDDPM")
    parser.add_argument("--input", required=True, help="Path to input HMDA CSV file")
    parser.add_argument("--output", required=True, help="Path to output synthetic CSV file")
    parser.add_argument("--model", required=True, choices=["ctgan", "tabddpm"], help="Generation model")
    parser.add_argument("--samples", type=int, default=None, help="Number of synthetic rows")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    source = _read_csv(args.input)
    synthetic = generate_synthetic_data(
        source,
        model=args.model,
        num_samples=args.samples,
        random_state=args.seed,
    )
    _write_csv(args.output, synthetic)


if __name__ == "__main__":
    main()
