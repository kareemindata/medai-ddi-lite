from __future__ import annotations

import numpy as np

from . import FP_BITS, FP_RADIUS


def smiles_to_fingerprint(smiles: str, radius: int = FP_RADIUS, n_bits: int = FP_BITS) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr
