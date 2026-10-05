"""Load audited definitions from the user-specified original inference architecture.
No notebook execution, global weight loading, fallback descriptors, or main().
"""
from pathlib import Path
import ast,random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Descriptors,rdmolops
ROOT=Path(__file__).resolve().parent.parent

def load_original():
 source=(ROOT/'training/original_infer.py').read_text()
 tree=ast.parse(source)
 names={'create_exact_training_vocab','get_backbone_sidechain_features','SimpleAttention',
        'SimpleMolFormerCNNLSTM'}
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {
      'CNN_CONFIGS','LSTM_HIDDEN_SIZE','LSTM_NUM_LAYERS','LSTM_BIDIRECTIONAL',
      'FC_LAYERS_SIZES','BACKBONE_SIDECHAIN_FEATURE_NAMES','SMILES_EMBEDDING_DIM'} for t in n.targets):
   nodes.append(n)
 ns=dict(np=np,pd=pd,torch=torch,nn=nn,F=F,Chem=Chem,Descriptors=Descriptors,rdmolops=rdmolops)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_inference_definitions','exec'),ns)
 return ns

ORIGINAL=load_original()
VARIANTS={'full':(True,True,True,True,True),
 'without_molformer':(True,True,True,False,True),
 'without_descriptors':(True,True,False,True,True),
 'without_cnn':(False,True,True,True,True),
 'without_lstm':(True,False,True,True,True),
 'without_augmentation':(True,True,True,True,False),
 'only_smiles':(True,True,False,False,True)}

class AblationModel(ORIGINAL['SimpleMolFormerCNNLSTM']):
 def __init__(self,variant,vocab_size,dropout=.05):
  super().__init__(vocab_size=vocab_size,embedding_dim=256,cnn_configs=ORIGINAL['CNN_CONFIGS'],
   lstm_hidden_size=128,lstm_num_layers=2,lstm_bidirectional=True,
   fc_layers_sizes=[512,256,128],num_backbone_features=17,num_molformer_features=64,dropout_rate=dropout)
  self.variant=variant
  self.use_cnn,self.use_lstm,self.use_descriptors,self.use_molformer,self.augment=VARIANTS[variant]
  dim=0
  if self.use_cnn:dim+=704
  else:del self.cnn_layers
  if self.use_lstm:dim+=256
  else:del self.lstm;del self.attention
  if self.use_descriptors:dim+=64
  else:del self.backbone_processor
  if self.use_molformer:dim+=32
  else:del self.molformer_processor
  if dim!=1056:self.fc_layers[0]=nn.Linear(dim,512)

 def forward(self,x,backbone_features,molformer_features):
  embedded=self.embedding_dropout(self.embedding(x));features=[]
  if self.use_cnn:
   cnn_input=embedded.transpose(1,2);parts=[]
   for layer in self.cnn_layers:
    y=F.relu(layer(cnn_input));parts.extend([self.global_max_pool(y).squeeze(-1),self.global_avg_pool(y).squeeze(-1)])
   features.append(torch.cat(parts,dim=1))
  if self.use_lstm:
   lstm,_=self.lstm(embedded);attended,_=self.attention(self.lstm_dropout(lstm),self.create_padding_mask(x))
   features.append(attended)
  if self.use_descriptors:features.append(self.backbone_processor(backbone_features))
  if self.use_molformer:features.append(self.molformer_processor(molformer_features))
  y=torch.cat(features,dim=1)
  for fc,ln,dropout in zip(self.fc_layers,self.fc_layer_norms,self.fc_dropouts):y=dropout(F.relu(ln(fc(y))))
  return self.output_layer(y).squeeze(1)


def seed_everything(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.set_num_threads(4)
