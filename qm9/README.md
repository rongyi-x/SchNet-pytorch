---
dataset_info:
  features:
  - name: num_atoms
    dtype: int64
  - name: atomic_symbols
    sequence: string
  - name: pos
    sequence:
      sequence: float64
  - name: charges
    sequence: float64
  - name: harmonic_oscillator_frequencies
    sequence: float64
  - name: smiles
    dtype: string
  - name: inchi
    dtype: string
  - name: A
    dtype: float64
  - name: B
    dtype: float64
  - name: C
    dtype: float64
  - name: mu
    dtype: float64
  - name: alpha
    dtype: float64
  - name: homo
    dtype: float64
  - name: lumo
    dtype: float64
  - name: gap
    dtype: float64
  - name: r2
    dtype: float64
  - name: zpve
    dtype: float64
  - name: u0
    dtype: float64
  - name: u
    dtype: float64
  - name: h
    dtype: float64
  - name: g
    dtype: float64
  - name: cv
    dtype: float64
  - name: canonical_smiles
    dtype: string
  - name: logP
    dtype: float64
  - name: qed
    dtype: float64
  - name: np_score
    dtype: float64
  - name: sa_score
    dtype: float64
  - name: ring_count
    dtype: int64
  - name: R3
    dtype: int64
  - name: R4
    dtype: int64
  - name: R5
    dtype: int64
  - name: R6
    dtype: int64
  - name: R7
    dtype: int64
  - name: R8
    dtype: int64
  - name: R9
    dtype: int64
  - name: single_bond
    dtype: int64
  - name: double_bond
    dtype: int64
  - name: triple_bond
    dtype: int64
  - name: aromatic_bond
    dtype: int64
  splits:
  - name: train
    num_bytes: 199395693
    num_examples: 133885
  download_size: 180380355
  dataset_size: 199395693
---
# Dataset Card for "QM9"

QM9 dataset from [Ruddigkeit et al., 2012](https://pubs.acs.org/doi/full/10.1021/ci300415d);
[Ramakrishnan et al., 2014](https://www.nature.com/articles/sdata201422).

Original data downloaded from: http://quantum-machine.org/datasets.
Additional annotations (QED, logP, SA score, NP score, bond and ring counts) added using [`rdkit`](https://www.rdkit.org/docs/index.html) library.

## Quick start usage:
```python
from datasets import load_dataset

ds = load_dataset("yairschiff/qm9")

# Random train/test splits as recommended by:
#   https://moleculenet.org/datasets-1
test_size = 0.1
seed = 1
ds.train_test_split(test_size=test_size, seed=seed)

# Use `ds['canonical_smiles']` from `rdkit` as inputs.
```


## Full processing steps

```python
import os
import typing

import datasets
import numpy as np
import pandas as pd
import rdkit
import torch
from rdkit import Chem as rdChem
from rdkit.Chem import Crippen, QED
from rdkit.Contrib.NP_Score import npscorer
from rdkit.Contrib.SA_Score import sascorer
from tqdm.auto import tqdm

# TODO: Update to 2024.03.6 release when available instead of suppressing warning!
#  See: https://github.com/rdkit/rdkit/issues/7625#
rdkit.rdBase.DisableLog('rdApp.warning')

def parse_float(
    s: str
) -> float:
    """Parses floats potentially written as exponentiated values.
    
        Copied from https://www.kaggle.com/code/tawe141/extracting-data-from-qm9-xyz-files/code
    """
    try:
        return float(s)
    except ValueError:
        base, power = s.split('*^')
        return float(base) * 10**float(power)


def count_rings_and_bonds(
    mol: rdChem.Mol, max_ring_size: int = -1
) -> typing.Dict[str, int]:
    """Counts bond and ring (by type)."""
    
    # Counting rings
    ssr = rdChem.GetSymmSSSR(mol)
    ring_count = len(ssr)
    
    ring_sizes = {} if max_ring_size < 0 else {i: 0 for i in range(3, max_ring_size+1)}
    for ring in ssr:
        ring_size = len(ring)
        if ring_size not in ring_sizes:
            ring_sizes[ring_size] = 0
        ring_sizes[ring_size] += 1
    
    # Counting bond types
    bond_counts = {
        'single': 0,
        'double': 0,
        'triple': 0,
        'aromatic': 0
    }
    
    for bond in mol.GetBonds():
        if bond.GetIsAromatic():
            bond_counts['aromatic'] += 1
        elif bond.GetBondType() == rdChem.BondType.SINGLE:
            bond_counts['single'] += 1
        elif bond.GetBondType() == rdChem.BondType.DOUBLE:
            bond_counts['double'] += 1
        elif bond.GetBondType() == rdChem.BondType.TRIPLE:
            bond_counts['triple'] += 1
    result = {
        'ring_count': ring_count,
    }
    for k, v in ring_sizes.items():
        result[f"R{k}"] = v

    for k, v in bond_counts.items():
        result[f"{k}_bond"] = v
    return result


def parse_xyz(
    filename: str,
    max_ring_size: int = -1,
    npscorer_model: typing.Optional[dict] = None,
    array_format: str = 'np'
) -> typing.Dict[str, typing.Any]:
    """Parses QM9 specific xyz files. 
    
        See https://www.nature.com/articles/sdata201422/tables/2 for reference.
        Adapted from https://www.kaggle.com/code/tawe141/extracting-data-from-qm9-xyz-files/code
    """
    assert array_format in ['np', 'pt'], \
        f"Invalid array_format: `{array_format}` provided. Must be one of `np` (numpy.array), `pt` (torch.tensor)."
    
    num_atoms = 0
    scalar_properties = []
    atomic_symbols = []
    xyz = []
    charges = []
    harmonic_vibrational_frequencies = []
    smiles = ''
    inchi = ''
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                num_atoms = int(line)
            elif line_num == 1:
                scalar_properties = [float(i) for i in line.split()[2:]]
            elif 2 <= line_num <= 1 + num_atoms:
                atom_symbol, x, y, z, charge = line.split()
                atomic_symbols.append(atom_symbol)
                xyz.append([parse_float(x), parse_float(y), parse_float(z)])
                charges.append(parse_float(charge))
            elif line_num == num_atoms + 2:
                harmonic_vibrational_frequencies = [float(i) for i in line.split()]
            elif line_num == num_atoms + 3:
                smiles = line.split()[0]
            elif line_num == num_atoms + 4:
                inchi = line.split()[0]

    array_wrap = np.array if array_format == 'np' else torch.tensor
    result = {
        'num_atoms': num_atoms,
        'atomic_symbols': atomic_symbols,
        'pos': array_wrap(xyz),
        'charges': array_wrap(charges),
        'harmonic_oscillator_frequencies': array_wrap(harmonic_vibrational_frequencies),
        'smiles': smiles,
        'inchi': inchi
    }
    scalar_property_labels = [
        'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u', 'h', 'g', 'cv'
    ]    
    scalar_properties = dict(zip(scalar_property_labels, scalar_properties))
    result.update(scalar_properties)

    # RdKit
    result['canonical_smiles'] = rdChem.CanonSmiles(result['smiles'])
    m = rdChem.MolFromSmiles(result['canonical_smiles'])
    result['logP'] = Crippen.MolLogP(m)
    result['qed'] = QED.qed(m)
    if npscorer_model is not None:
        result['np_score'] = npscorer.scoreMol(m, npscorer_model)
    result['sa_score'] = sascorer.calculateScore(m)
    result.update(count_rings_and_bonds(m, max_ring_size=max_ring_size))
    
    return result

"""
    Download xyz files from:
        https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
    > wget https://figshare.com/ndownloader/files/3195389/dsgdb9nsd.xyz.tar.bz2
    > mkdir dsgdb9nsd.xyz
    > tar -xvjf dsgdb9nsd.xyz.tar.bz2 -C dsgdb9nsd.xyz
"""
MAX_RING_SIZE = 9
fscore = npscorer.readNPModel()
xyz_dir_path = '<PATH TO dsgdb9nsd.xyz>'
parsed_xyz = []
for file in tqdm(sorted(os.listdir(xyz_dir_path)), desc='Parsing'):
    parsed = parse_xyz(os.path.join(xyz_dir_path, file),
                       max_ring_size=MAX_RING_SIZE,
                       npscorer_model=fscore,
                       array_format='np')
    parsed_xyz.append(parsed)

qm9_df = pd.DataFrame(data=parsed_xyz)

# Conversion below is needed to avoid:
#   `ArrowInvalid: ('Can only convert 1-dimensional array values',
#   'Conversion failed for column pos with type object')`
qm9_df['pos'] = qm9_df['pos'].apply(lambda x: [xi for xi in x])

dataset = datasets.Dataset.from_pandas(qm9_df)
```