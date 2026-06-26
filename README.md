# medai-ddi-lite

A **lightweight Drug-Drug Interaction (DDI) classifier** that pairs Morgan
molecular fingerprints with a gradient-boosted model to predict whether two
drugs interact.

> **Companion repo.** This is the lightweight, fingerprint-only counterpart to
> the in-progress flagship work in
> [kareemindata/ddi-risk-explorer](https://github.com/kareemindata/ddi-risk-explorer).
> It exists to give a clean, reproducible baseline that anyone can run on a
> laptop in minutes.

> **Disclaimer.** Educational / research use only. Not for clinical
> decision-making.

## What it does

- Takes a CSV of drug pairs with SMILES + binary `interacts` label.
- Generates Morgan fingerprints (radius 2, 2048 bits) per drug via RDKit.
- Combines two fingerprints into a single feature vector (concat + XOR).
- Trains an XGBoost classifier with stratified 5-fold CV.
- Reports ROC-AUC, PR-AUC, F1, and confusion matrix.
- Exposes a `predict_pair(smiles_a, smiles_b)` API.

## Dataset

Expects a CSV with columns:

```
drug_a, drug_b, smiles_a, smiles_b, interacts
```

`interacts` is `0` or `1`. The `BIOSNAP` ChCh-Miner edges or DrugBank pairs
(with negative sampling) both work. The `data/` folder is gitignored.

You can generate a small toy CSV for smoke-testing:

```bash
python -m src.toy_data --out data/toy.csv --n 200
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.train --data data/pairs.csv --out runs/baseline
python -m src.predict --model runs/baseline/best_model.joblib \
    --smiles-a "CC(=O)OC1=CC=CC=C1C(=O)O" \
    --smiles-b "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
```

## Project layout

```
src/
  fingerprints.py   # SMILES -> Morgan fingerprint
  features.py       # pair-feature combiners
  train.py          # XGBoost + stratified CV
  predict.py        # pair-inference CLI
  toy_data.py       # tiny synthetic dataset for smoke tests
tests/
  test_features.py
```

## Caveats

- Fingerprint baselines are weak compared to GNN / pretrained-chemistry
  models. This repo is intentionally a floor, not a ceiling.
- Random negatives over-estimate performance. For honest evaluation, use
  scaffold-split or time-split negatives.
- The flagship project [[ddi-risk-explorer]] handles structured drug
  metadata, IEEE-paper evaluation, and richer feature spaces.

## License

MIT
