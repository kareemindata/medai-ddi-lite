import argparse
import json

import joblib

from .features import combine_pair
from .fingerprints import smiles_to_fingerprint


def predict_pair(model, smiles_a: str, smiles_b: str) -> float:
    fp_a = smiles_to_fingerprint(smiles_a)
    fp_b = smiles_to_fingerprint(smiles_b)
    feat = combine_pair(fp_a, fp_b).reshape(1, -1)
    return float(model.predict_proba(feat)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--smiles-a", required=True)
    parser.add_argument("--smiles-b", required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    prob = predict_pair(artifact["model"], args.smiles_a, args.smiles_b)
    print(json.dumps({"prob_interacts": prob, "label_at_0.5": int(prob >= 0.5)}, indent=2))


if __name__ == "__main__":
    main()
