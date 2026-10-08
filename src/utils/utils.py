import math
import subprocess

import autokeras as ak
import pandas as pd
from keras.models import Model
from rdkit import Chem
from rdkit.Chem import Descriptors


def calc_mw(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return int(Descriptors.MolWt(mol))
    else:
        raise ValueError("Invalid smiles is input")


def create_result_csv(
    output_path: str,
    y_preds: float,
    y_true: float,
    true_column_name: str = "y_test",
) -> None:
    y_preds = pd.Series(y_preds)
    y_true = pd.Series(y_true)
    pd.concat([y_preds, y_true], axis=1).to_csv(
        output_path, header=["y_preds", true_column_name], index=False
    )


def list_gpu_names() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    gpu_names = result.stdout.strip().split("\n")
    return gpu_names
