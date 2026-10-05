"""Compatibility for legacy torchgraphs with current PyTorch.

Only sum aggregation is used by the author model. Replace compiled
torch_scatter.scatter_add with the equivalent native index_add operation.
"""
import sys, types
import torch

def install():
    mod=types.ModuleType('torch_scatter')
    def scatter_add(src,index,dim=0,out=None,dim_size=None):
        if dim!=0:raise NotImplementedError('This adapter only supports the author model dim=0')
        size=dim_size if dim_size is not None else (int(index.max())+1 if index.numel() else 0)
        result=src.new_zeros((size,)+src.shape[1:]) if out is None else out
        return result.index_add_(0,index,src)
    def scatter_mean(src,index,dim=0,out=None,dim_size=None):
        result=scatter_add(src,index,dim,out,dim_size)
        counts=scatter_add(src.new_ones((src.shape[0],)),index,dim,dim_size=result.shape[0]).clamp_min(1)
        return result/counts.reshape((-1,)+(1,)*(src.ndim-1))
    mod.scatter_add=scatter_add;mod.scatter_mean=scatter_mean
    sys.modules['torch_scatter']=mod
    from torch.utils.data._utils import collate
    if not hasattr(collate,'_use_shared_memory'):collate._use_shared_memory=False

def fix_pandas_assignments():
    # pandas 3 copy-on-write prevents two legacy DataFrame updates. Preserve
    # their intended in-place writes using .loc; leave downloaded source intact.
    import ast
    from pathlib import Path
    from purs.graph.pu_gn_exp.polymer_unit import PURS
    source=Path(PURS.__file__).read_text()
    source=source.replace('str_DataFrame[0][i]=j','str_DataFrame.loc[i,0]=j')
    source=source.replace('str_DataFrame[k+1][j]=n','str_DataFrame.loc[j,k+1]=n')
    tree=ast.parse(source)
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),PURS.__file__,'exec'),PURS.__dict__)
