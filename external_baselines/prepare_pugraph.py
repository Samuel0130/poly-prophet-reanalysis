"""Prepare graphs with the downloaded author team's actual PURS implementation."""
from pathlib import Path
import sys, json, time, contextlib, io
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external_baselines/vendor'))
from graph_compat import install, fix_pandas_assignments
install()
import pandas as pd
import torch
from purs.graph.legacy_purs_adapter import _build_graph_dist_for_rows
fix_pandas_assignments()

def main():
    cache=ROOT/'external_baselines/cache';cache.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/'training/artifacts/analysis_dataset.csv')
    path=cache/'pugraph.pt'
    graphs=torch.load(path,weights_only=False) if path.exists() else {}
    report_path=ROOT/'external_baselines/results/pugraph_conversion.csv'
    report=pd.read_csv(report_path).to_dict('records') if report_path.exists() else []
    done={int(r['row_id']) for r in report}
    for i,row in df.iterrows():
        if i in done:continue
        start=time.time();status='ok';reason='';nodes=0;edges=0
        try:
            # Per-row building avoids dataset-wide unit indices encoding held-out structures.
            with contextlib.redirect_stdout(io.StringIO()):
                g=_build_graph_dist_for_rows([dict(**{'Compound ID':str(i)},smiles=row.SMILES)])[str(i)]
            if g.num_nodes<1:raise ValueError('empty unit graph')
            # Isolated units have zero edges; represent with the known 8D empty tensor.
            if g.num_edges==0:g=g.evolve(edge_features=torch.empty((0,8),dtype=torch.float32))
            assert g.node_features.shape==(g.num_nodes,167)
            assert g.edge_features.shape==(g.num_edges,8)
            assert len(g.senders)==len(g.receivers)==g.num_edges
            graphs[i]=g;nodes=g.num_nodes;edges=g.num_edges
        except Exception as e:status='failed';reason=f'{type(e).__name__}: {e}'
        report.append(dict(row_id=i,status=status,nodes=nodes,edges=edges,reason=reason,seconds=time.time()-start))
        if (i+1)%100==0 or i==len(df)-1:
            torch.save(graphs,path);pd.DataFrame(report).to_csv(report_path,index=False)
            print('GRAPH',i+1,'/',len(df),'ok',len(graphs),'failed',sum(r['status']!='ok' for r in report),flush=True)

if __name__=='__main__':main()
