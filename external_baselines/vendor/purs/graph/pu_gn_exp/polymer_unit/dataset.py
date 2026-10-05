from pathlib import Path
from functools import lru_cache
from typing import Tuple

import torch
import pandas as pd
from pandas.api.types import CategoricalDtype

import torchgraphs as tg
try:
    from .compat import ensure_repo_root_on_path
except ImportError:
    from compat import ensure_repo_root_on_path

ensure_repo_root_on_path()
from purs.graph.legacy_purs_adapter import build_graph_dist_from_csv

symbols = CategoricalDtype([
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
    'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
    'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
], ordered=True)

bonds = CategoricalDtype([
    'SINGLE',
    'DOUBLE',
    'TRIPLE',
    'AROMATIC'
], ordered=True)

def get_pu_dist(file_name):
    return build_graph_dist_from_csv(file_name)

def smiles_to_graph(graph_dist,name):
    name = str(name)
    graph = graph_dist[name]
    return graph


@lru_cache(maxsize=8)
def load_graph_dist(path: str):
    return get_pu_dist(path)
   

class SolubilityDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        self.path = path
        self.graph_dist = load_graph_dist(path.as_posix())
        self.df = pd.read_csv(path)
        self.df = self.df[self.df['Compound ID'].astype(str).isin(self.graph_dist)].reset_index(drop=True)
        # self.df['molecules'] = self.df.smiles.apply(smiles_to_graph)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item) -> Tuple[tg.Graph, float]:
     
        mol = smiles_to_graph(self.graph_dist,self.df['Compound ID'].iloc[item])
        target = self.df['PCE_max'].iloc[item]
        return mol, torch.tensor(float(target), dtype=torch.float32)


def describe(cfg):
    pd.options.display.precision = 2
    pd.options.display.max_columns = 999
    pd.options.display.expand_frame_repr = False
    target = Path(cfg.target).expanduser().resolve()
    if target.is_dir():
        paths = target.glob('*.pt')
    else:
        paths = [target]
    for p in paths:
        print(f"Loading dataset from: {p}")
        dataset = SolubilityDataset(p)
        print(f"{p.with_suffix('').name.capitalize()} contains:\n"
              f"{dataset.df.drop(columns=['molecules'], errors='ignore').describe().transpose()}")


def main():
    from argparse import ArgumentParser
    try:
        from .config import Config
    except ImportError:
        from config import Config

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    sp_print = subparsers.add_parser('print', help='Print parsed configuration')
    sp_print.add_argument('config', nargs='*')
    sp_print.set_defaults(command=lambda c: print(c.toYAML()))

    sp_describe = subparsers.add_parser('describe', help='Describe existing datasets')
    sp_describe.add_argument('config', nargs='*')
    sp_describe.set_defaults(command=describe)

    args = parser.parse_args()
    cfg = Config.build(*args.config)
    args.command(cfg)


if __name__ == '__main__':
    main()
