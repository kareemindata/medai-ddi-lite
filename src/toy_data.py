"""Generate a tiny synthetic DDI CSV for smoke testing.

Picks pairs from a small SMILES seed list, labels them based on a deterministic
rule (whether the two fingerprints share enough bits) so the dataset is
learnable but obviously synthetic.
"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np

from .fingerprints import smiles_to_fingerprint

SEED_SMILES = {
    "aspirin":     "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine":    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen":   "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "warfarin":    "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "metformin":   "CN(C)C(=N)NC(=N)N",
    "atorvastatin":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "lisinopril":  "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    names = list(SEED_SMILES.keys())
    fps = {n: smiles_to_fingerprint(SEED_SMILES[n]) for n in names}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["drug_a", "drug_b", "smiles_a", "smiles_b", "interacts"])
        for _ in range(args.n):
            a, b = rng.sample(names, 2)
            shared = int(np.bitwise_and(fps[a], fps[b]).sum())
            label = 1 if shared > 90 else 0
            w.writerow([a, b, SEED_SMILES[a], SEED_SMILES[b], label])
    print(f"wrote {args.n} pairs -> {args.out}")


if __name__ == "__main__":
    main()
