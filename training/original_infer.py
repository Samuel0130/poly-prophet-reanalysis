import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
import warnings
import random
from tqdm import tqdm
import math
import statistics as stat
import pickle
import json

# 添加MolFormer路径
MOLFOLD_PATH = '/kaggle/input/molfold'
sys.path.append(MOLFOLD_PATH)
sys.path.append(os.path.join(MOLFOLD_PATH, 'molformer-xl-local'))

# 加载MolFormer
MOLFORMER_AVAILABLE = False
molformer_model = None
molformer_tokenizer = None

try:
    from transformers import AutoModel, AutoTokenizer
    model_path = os.path.join(MOLFOLD_PATH, 'molformer-xl-local')
    
    molformer_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    molformer_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    molformer_model = molformer_model.to(device)
    molformer_model.eval()
    
    MOLFORMER_AVAILABLE = True
    print("✓ MolFormer loaded successfully!")
    
except Exception as e:
    print(f"Failed to load MolFormer: {e}")
    MOLFORMER_AVAILABLE = False

warnings.filterwarnings('ignore')

# ===============================
# 配置和超参数
# ===============================
SEED = 42
TARGET_COLUMNS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预训练模型路径
TRAINED_MODELS_PATH = '/kaggle/input/cnnlstm-bamod4c-26793f-output/trained_models/'

# 数据路径
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'

# 保持原有模型参数
SMILES_EMBEDDING_DIM = 256
DROPOUT_RATE = 0.05
BATCH_SIZE = 16

# 目标变量缩放因子
TARGET_SCALING = {
    'FFV': 1000,
    'Tc': 100,
    'Tg': 1,
    'Density': 1,
    'Rg': 1
}

# 保持原有CNN配置
CNN_CONFIGS = [
    {'out_channels': 64, 'kernel_size': 3, 'dilation': 1},
    {'out_channels': 64, 'kernel_size': 5, 'dilation': 1},
    {'out_channels': 64, 'kernel_size': 7, 'dilation': 1},
    {'out_channels': 64, 'kernel_size': 3, 'dilation': 2},
    {'out_channels': 64, 'kernel_size': 5, 'dilation': 2},
    {'out_channels': 32, 'kernel_size': 15, 'dilation': 1},
]

LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
FC_LAYERS_SIZES = [512, 256, 128]

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clean_and_validate_smiles(smiles):
    """清洗和验证SMILES"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    bad_patterns = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]']
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except:
        return None

# ===============================
# 使用训练时的确切词汇表
# ===============================

def create_exact_training_vocab():
    """使用训练时的确切字符列表创建词汇表"""
    # 这是您训练时使用的确切字符列表
    training_chars = [
        '#', '%', '(', ')', '*', '+', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        '=', '@', 'B', 'C', 'F', 'G', 'H', 'N', 'O', 'P', 'S', 'T', '[', '\\', ']', 
        'a', 'c', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's'
    ]
    
    # 创建字符到索引的映射
    char_dict = {char: i + 1 for i, char in enumerate(training_chars)}
    char_dict['&'] = 0  # padding token
    char_dict['!'] = len(char_dict)  # unknown token
    
    print(f"Created exact training vocab with size: {len(char_dict)}")
    print(f"Training chars: {training_chars}")
    
    return char_dict

# ===============================
# MolFormer特征提取器
# ===============================

class SimpleMolFormerExtractor:
    """MolFormer特征提取器"""
    
    def __init__(self):
        self.model = molformer_model
        self.tokenizer = molformer_tokenizer
        self.is_available = MOLFORMER_AVAILABLE
        self.device = DEVICE
        self.feature_cache = {}
        
        # 固定的特征维度
        self.molformer_dim = 64  # 固定64维
        
        if self.is_available:
            print("✓ Simple MolFormer extractor initialized")
    
    def get_molformer_features(self, smiles_list):
        """获取MolFormer特征"""
        if not self.is_available:
            return self._get_fallback_features(smiles_list)
        
        try:
            # 批量处理
            if len(smiles_list) > 32:
                all_features = []
                for i in range(0, len(smiles_list), 32):
                    batch_smiles = smiles_list[i:i+32]
                    batch_features = self._extract_batch_features(batch_smiles)
                    all_features.append(batch_features)
                result = np.vstack(all_features)
            else:
                result = self._extract_batch_features(smiles_list)
            
            # 全局NaN检查和清理
            if np.isnan(result).any() or np.isinf(result).any():
                print("Warning: NaN/Inf found in MolFormer features, cleaning...")
                result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return result
                
        except Exception as e:
            print(f"Error in MolFormer feature extraction: {e}")
            return self._get_fallback_features(smiles_list)
    
    def _extract_batch_features(self, smiles_list):
        """批量提取特征"""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    smiles_list,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取最后一层隐藏状态
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                
                # 简单的CLS token + 平均池化
                cls_features = last_hidden_state[:, 0, :]  # CLS token
                mean_features = last_hidden_state[:, 1:, :].mean(dim=1)  # 平均池化
                
                # 组合特征
                combined_features = (cls_features + mean_features) / 2
                
                # 检查组合特征中的NaN
                if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
                    print("Warning: NaN/Inf in combined features, cleaning...")
                    combined_features = torch.nan_to_num(combined_features, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 降维到固定维度
                if combined_features.shape[1] > self.molformer_dim:
                    # 简单选择前N维
                    reduced_features = combined_features[:, :self.molformer_dim]
                else:
                    # 填充到目标维度
                    padding = torch.zeros(combined_features.shape[0], 
                                        self.molformer_dim - combined_features.shape[1]).to(self.device)
                    reduced_features = torch.cat([combined_features, padding], dim=1)
                
                result = reduced_features.cpu().numpy()
                
                # 最终检查
                if np.isnan(result).any() or np.isinf(result).any():
                    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
                
                return result
                
        except Exception as e:
            print(f"Error in batch feature extraction: {e}")
            return self._get_fallback_features(smiles_list)
    
    def _get_fallback_features(self, smiles_list):
        """改进的备选特征 - 更好的NaN处理"""
        features = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    features.append([0.0] * self.molformer_dim)
                    continue
                
                # 计算基础描述符，增强NaN检查
                raw_features = []
                try:
                    raw_features = [
                        Descriptors.MolWt(mol),
                        Descriptors.NumHDonors(mol),
                        Descriptors.NumHAcceptors(mol),
                        Descriptors.TPSA(mol),
                        Descriptors.NumRotatableBonds(mol),
                        Descriptors.BertzCT(mol),
                        Descriptors.NumAromaticRings(mol),
                        Descriptors.NumSaturatedRings(mol),
                        Descriptors.FractionCsp3(mol),
                        Descriptors.Chi0v(mol),
                        mol.GetNumBonds() / max(mol.GetNumAtoms(), 1),
                        mol.GetNumAtoms(),
                        len(smiles),
                        smiles.count('*'),
                        len([c for c in smiles if c in '()[]'])
                    ]
                except:
                    raw_features = [0.0] * 15
                
                # 严格的NaN/Inf检查和清理
                clean_features = []
                for f in raw_features:
                    if f is None or np.isnan(f) or np.isinf(f):
                        clean_features.append(0.0)
                    else:
                        clean_features.append(float(f))
                
                # 填充或截断到目标维度
                if len(clean_features) >= self.molformer_dim:
                    final_features = clean_features[:self.molformer_dim]
                else:
                    final_features = clean_features + [0.0] * (self.molformer_dim - len(clean_features))
                
                # 最终检查确保没有NaN
                final_features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in final_features]
                features.append(final_features)
                        
            except Exception as e:
                features.append([0.0] * self.molformer_dim)
        
        result = np.array(features, dtype=np.float32)
        
        # 全局NaN检查
        if np.isnan(result).any() or np.isinf(result).any():
            print("Warning: NaN/Inf found in fallback features, replacing with zeros")
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
        return result

# 全局特征提取器
molformer_extractor = SimpleMolFormerExtractor()

# ===============================
# 主链/侧链特征提取
# ===============================

def get_backbone_sidechain_features(smiles: str) -> dict:
    """改进的主链/侧链特征提取 - 更好的错误处理"""
    feature_keys = [
        'bb_atom_count', 'sc_atom_count', 'sc_to_bb_atom_ratio',
        'bb_mw', 'sc_mw', 'sc_to_bb_mw_ratio',
        'bb_rotatable_bonds', 'sc_rotatable_bonds',
        'bb_aromatic_rings', 'sc_aromatic_rings',
        'bb_tpsa', 'sc_tpsa',
        'avg_sc_fragment_size', 'avg_sc_fragment_mw',
        'num_sc_fragments', "stddev_sc_fragment_size", "stddev_sc_fragment_mw"
    ]
    
    # 初始化为0.0而不是NaN
    features = {key: 0.0 for key in feature_keys}

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return features

        star_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        if len(star_atoms) != 2:
            return features

        if not star_atoms[0].GetNeighbors() or not star_atoms[1].GetNeighbors():
            return features
            
        endpoint1 = star_atoms[0].GetNeighbors()[0].GetIdx()
        endpoint2 = star_atoms[1].GetNeighbors()[0].GetIdx()

        if endpoint1 == endpoint2:
            features['bb_atom_count'] = 1.0
            features['sc_atom_count'] = 0.0
            return features

        backbone_path = rdmolops.GetShortestPath(mol, endpoint1, endpoint2)
        if not backbone_path:
            return features
        
        backbone_atom_indices = set(backbone_path)
        all_real_atom_indices = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0}
        sidechain_atom_indices = all_real_atom_indices - backbone_atom_indices

        features['bb_atom_count'] = float(len(backbone_atom_indices))
        features['sc_atom_count'] = float(len(sidechain_atom_indices))
        
        if features['bb_atom_count'] > 0:
            features['sc_to_bb_atom_ratio'] = features['sc_atom_count'] / features['bb_atom_count']
        
        try:
            bb_mw = sum(mol.GetAtomWithIdx(i).GetMass() for i in backbone_atom_indices)
            sc_mw = sum(mol.GetAtomWithIdx(i).GetMass() for i in sidechain_atom_indices)
            
            features['bb_mw'] = float(bb_mw) if not np.isnan(bb_mw) else 0.0
            features['sc_mw'] = float(sc_mw) if not np.isnan(sc_mw) else 0.0
            
            if bb_mw > 0:
                features['sc_to_bb_mw_ratio'] = float(sc_mw / bb_mw)
        except:
            pass
            
    except Exception as e:
        pass
        
    # 最终检查所有特征值
    for key in features:
        if features[key] is None or np.isnan(features[key]) or np.isinf(features[key]):
            features[key] = 0.0
        else:
            features[key] = float(features[key])
        
    return features

def add_backbone_sidechain_features_to_df(df):
    """将主链和侧链特征计算并添加到DataFrame中"""
    features = df['SMILES'].apply(get_backbone_sidechain_features)
    feature_df = pd.json_normalize(features)
    feature_df = feature_df.fillna(0.0)
    
    # 额外的NaN清理
    for col in feature_df.columns:
        feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0.0)
    
    return pd.concat([df, feature_df], axis=1)

BACKBONE_SIDECHAIN_FEATURE_NAMES = [
    'bb_atom_count', 'sc_atom_count', 'sc_to_bb_atom_ratio',
    'bb_mw', 'sc_mw', 'sc_to_bb_mw_ratio',
    'bb_rotatable_bonds', 'sc_rotatable_bonds',
    'bb_aromatic_rings', 'sc_aromatic_rings',
    'bb_tpsa', 'sc_tpsa',
    'avg_sc_fragment_size', 'avg_sc_fragment_mw',
    'num_sc_fragments', "stddev_sc_fragment_size", "stddev_sc_fragment_mw"
]

# ===============================
# 数据集类 - 修复了tokenization问题
# ===============================

class SimpleMolFormerDataset(Dataset):
    def __init__(self, smiles_list, targets, backbone_features, molformer_features, 
                 char_dict, max_len, is_training=False):
        self.smiles_list = smiles_list
        self.targets = targets
        self.backbone_features = backbone_features
        self.molformer_features = molformer_features
        self.char_dict = char_dict
        self.max_len = max_len
        self.pad_token_id = char_dict['&']
        self.unknown_token_id = char_dict['!']
        self.is_training = is_training
        
        # 预处理tokenization以避免运行时错误
        self.tokenized_smiles = []
        for smiles in smiles_list:
            tokenized = self._safe_tokenize(smiles)
            self.tokenized_smiles.append(tokenized)

    def _safe_tokenize(self, smiles):
        """安全的tokenization，处理未知字符"""
        tokenized = []
        for char in smiles:
            if char in self.char_dict:
                tokenized.append(self.char_dict[char])
            else:
                tokenized.append(self.unknown_token_id)  # 使用未知token
        
        # 填充或截断
        if len(tokenized) < self.max_len:
            tokenized = tokenized + [self.pad_token_id] * (self.max_len - len(tokenized))
        else:
            tokenized = tokenized[:self.max_len]
        
        return tokenized

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        target = self.targets[idx] if self.targets is not None else 0.0
        backbone_feats = self.backbone_features[idx]
        molformer_feats = self.molformer_features[idx]
        tokenized_smiles = self.tokenized_smiles[idx]
        
        # 确保特征没有NaN
        backbone_feats = np.nan_to_num(backbone_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        molformer_feats = np.nan_to_num(molformer_feats, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return (torch.tensor(tokenized_smiles, dtype=torch.long),
                torch.tensor(backbone_feats, dtype=torch.float),
                torch.tensor(molformer_feats, dtype=torch.float),
                torch.tensor(target, dtype=torch.float))

# ===============================
# 注意力机制和模型定义
# ===============================

class SimpleAttention(nn.Module):
    """简化的注意力机制"""
    def __init__(self, hidden_size, dropout_rate=0.1):
        super(SimpleAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, lstm_output, mask=None):
        attention_scores = self.attention_weights(lstm_output)
        attention_scores = attention_scores.squeeze(-1)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        weighted_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_output, attention_weights

class SimpleMolFormerCNNLSTM(nn.Module):
    """CNN-LSTM模型"""
    def __init__(self, vocab_size, embedding_dim, cnn_configs, lstm_hidden_size, 
                 lstm_num_layers, lstm_bidirectional, fc_layers_sizes, 
                 num_backbone_features, num_molformer_features, dropout_rate=0.3):
        super(SimpleMolFormerCNNLSTM, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # CNN分支
        self.cnn_layers = nn.ModuleList()
        total_cnn_out_channels = 0
        
        for config in cnn_configs:
            self.cnn_layers.append(nn.Conv1d(
                embedding_dim, config['out_channels'], 
                config['kernel_size'], 
                dilation=config.get('dilation', 1),
                padding='same'
            ))
            total_cnn_out_channels += config['out_channels']
        
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=dropout_rate if lstm_num_layers > 1 else 0
        )
        
        lstm_output_size = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # 注意力机制
        self.attention = SimpleAttention(lstm_output_size, dropout_rate)
        
        # 主链/侧链特征处理器
        backbone_processor_output_size = 64
        self.backbone_processor = nn.Sequential(
            nn.Linear(num_backbone_features, backbone_processor_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # MolFormer特征处理器
        molformer_processor_output_size = 32
        self.molformer_processor = nn.Sequential(
            nn.Linear(num_molformer_features, molformer_processor_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 全连接层输入维度
        cnn_feature_size = total_cnn_out_channels * 2  # max + avg pooling
        lstm_feature_size = lstm_output_size
        combined_input_size = (cnn_feature_size + lstm_feature_size + 
                             backbone_processor_output_size + molformer_processor_output_size)
        
        # 全连接层
        self.fc_layers = nn.ModuleList()
        self.fc_layer_norms = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        
        prev_size = combined_input_size
        for layer_size in fc_layers_sizes:
            self.fc_layers.append(nn.Linear(prev_size, layer_size))
            self.fc_layer_norms.append(nn.LayerNorm(layer_size))
            self.fc_dropouts.append(nn.Dropout(dropout_rate))
            prev_size = layer_size
        
        # 输出层
        self.output_layer = nn.Linear(fc_layers_sizes[-1], 1)

    def create_padding_mask(self, x, pad_token_id=0):
        """创建padding mask"""
        return (x != pad_token_id).float()

    def forward(self, x, backbone_features, molformer_features):
        batch_size = x.size(0)
        
        # 输入验证和NaN检查
        if torch.isnan(backbone_features).any() or torch.isinf(backbone_features).any():
            backbone_features = torch.nan_to_num(backbone_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(molformer_features).any() or torch.isinf(molformer_features).any():
            molformer_features = torch.nan_to_num(molformer_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 创建padding mask
        padding_mask = self.create_padding_mask(x)
        
        # 嵌入层
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        if torch.isnan(embedded).any():
            embedded = torch.nan_to_num(embedded, nan=0.0)
        
        # CNN分支
        cnn_input = embedded.transpose(1, 2)
        cnn_features = []
        
        for cnn_layer in self.cnn_layers:
            cnn_output = F.relu(cnn_layer(cnn_input))
            if torch.isnan(cnn_output).any():
                cnn_output = torch.nan_to_num(cnn_output, nan=0.0)
            
            max_pooled = self.global_max_pool(cnn_output).squeeze(-1)
            avg_pooled = self.global_avg_pool(cnn_output).squeeze(-1)
            cnn_features.extend([max_pooled, avg_pooled])
        
        cnn_combined = torch.cat(cnn_features, dim=1)
        
        # LSTM分支
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)
        
        if torch.isnan(lstm_out).any():
            lstm_out = torch.nan_to_num(lstm_out, nan=0.0)
        
        # 注意力机制
        attn_output, attn_weights = self.attention(lstm_out, padding_mask)
        
        # 处理特征
        processed_backbone_features = self.backbone_processor(backbone_features)
        processed_molformer_features = self.molformer_processor(molformer_features)
        
        # 特征融合前的最终检查
        features_to_check = [cnn_combined, attn_output, processed_backbone_features, processed_molformer_features]
        cleaned_features = []
        
        for feature in features_to_check:
            if torch.isnan(feature).any() or torch.isinf(feature).any():
                feature = torch.nan_to_num(feature, nan=0.0, posinf=1e6, neginf=-1e6)
            cleaned_features.append(feature)
        
        # 特征融合
        combined_features = torch.cat(cleaned_features, dim=1)
        
        # 全连接层
        x = combined_features
        for fc, ln, dropout in zip(self.fc_layers, self.fc_layer_norms, self.fc_dropouts):
            x = fc(x)
            x = ln(x)
            x = torch.relu(x)
            x = dropout(x)
            
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        
        # 输出层
        output = self.output_layer(x).squeeze(1)
        
        # 最终输出检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return output

# ===============================
# 工具函数
# ===============================

def get_model_config(target_col):
    """根据目标变量获取特定的模型配置"""
    if target_col in ['FFV', 'Tc']:
        return {
            'dropout_rate': 0.05,
        }
    else:
        return {
            'dropout_rate': DROPOUT_RATE,
        }

def scale_target(values, target_col, inverse=False):
    """缩放或逆缩放目标变量"""
    scale_factor = TARGET_SCALING.get(target_col, 1)
    if inverse:
        return values / scale_factor
    else:
        return values * scale_factor

# ===============================
# 主预测函数
# ===============================

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"MolFormer available: {MOLFORMER_AVAILABLE}")
    
    # 加载测试数据
    try:
        test_df = pd.read_csv(BASE_PATH + 'test.csv')
        print(f"Loaded test data with {len(test_df)} samples")
        print("Test data columns:", test_df.columns.tolist())
        print("Test data sample:")
        print(test_df.head())
    except FileNotFoundError:
        print(f"Error: Test data file not found at {BASE_PATH}")
        return
    
    # 清洗SMILES
    if 'SMILES' not in test_df.columns:
        print("Error: 'SMILES' column not found in test data")
        return
    
    print("Cleaning SMILES...")
    test_df['SMILES'] = test_df['SMILES'].apply(clean_and_validate_smiles)
    test_df = test_df[test_df['SMILES'].notnull()].reset_index(drop=True)
    print(f"After cleaning: {len(test_df)} valid samples")
    
    # 添加主链/侧链特征
    print("Calculating backbone/sidechain features...")
    test_df = add_backbone_sidechain_features_to_df(test_df)
    
    # 提取MolFormer特征
    print("Extracting MolFormer features...")
    test_molformer_features = molformer_extractor.get_molformer_features(test_df['SMILES'].tolist())
    print(f"MolFormer features shape: {test_molformer_features.shape}")
    
    # 创建与训练时完全一致的词汇表
    print("Creating exact training vocabulary...")
    training_vocab = create_exact_training_vocab()
    max_smiles_len = max(len(smiles) for smiles in test_df['SMILES']) + 20
    
    print(f"Training vocab size: {len(training_vocab)}")
    print(f"Max SMILES length: {max_smiles_len}")
    
    # 初始化结果DataFrame
    if 'id' in test_df.columns:
        submission_df = pd.DataFrame({'id': test_df['id']})
    else:
        submission_df = pd.DataFrame({'id': range(len(test_df))})
    
    # 为每个目标进行预测
    available_targets = []
    
    for target_col in TARGET_COLUMNS:
        print(f"\n{'='*50}")
        print(f"Predicting for: {target_col}")
        print(f"{'='*50}")
        
        # 检查是否有对应的模型文件
        model_files = []
        for fold in range(1, 6):  # 假设有5折
            model_file = f"{TRAINED_MODELS_PATH}model_{target_col}_fold_{fold}.pth"
            if os.path.exists(model_file):
                model_files.append(model_file)
        
        if not model_files:
            print(f"No trained models found for {target_col}")
            submission_df[target_col] = 0.0
            continue
        
        print(f"Found {len(model_files)} model files for {target_col}")
        available_targets.append(target_col)
        
        fold_predictions = []
        
        for model_file in model_files:
            try:
                print(f"Loading model from: {model_file}")
                
                # 加载模型配置
                model_config = get_model_config(target_col)
                
                # 使用训练时的确切词汇表大小创建模型
                model = SimpleMolFormerCNNLSTM(
                    vocab_size=len(training_vocab),
                    embedding_dim=SMILES_EMBEDDING_DIM,
                    cnn_configs=CNN_CONFIGS,
                    lstm_hidden_size=LSTM_HIDDEN_SIZE,
                    lstm_num_layers=LSTM_NUM_LAYERS,
                    lstm_bidirectional=LSTM_BIDIRECTIONAL,
                    fc_layers_sizes=FC_LAYERS_SIZES,
                    num_backbone_features=len(BACKBONE_SIDECHAIN_FEATURE_NAMES),
                    num_molformer_features=molformer_extractor.molformer_dim,
                    dropout_rate=model_config['dropout_rate']
                )
                
                # 加载预训练权重
                checkpoint = torch.load(model_file, map_location=DEVICE)
                model.load_state_dict(checkpoint)
                model.to(DEVICE)
                model.eval()
                print("✓ Model loaded successfully")
                
                # 创建测试数据集
                test_dataset = SimpleMolFormerDataset(
                    test_df['SMILES'].values, 
                    None,  # no targets for test data
                    test_df[BACKBONE_SIDECHAIN_FEATURE_NAMES].values,
                    test_molformer_features,
                    training_vocab, max_smiles_len,
                    is_training=False
                )
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
                
                # 预测
                predictions = []
                with torch.no_grad():
                    for smiles, backbone_features, molformer_features, _ in tqdm(test_loader, desc=f"Predicting"):
                        smiles = smiles.to(DEVICE)
                        backbone_features = backbone_features.to(DEVICE)
                        molformer_features = molformer_features.to(DEVICE)
                        
                        pred = model(smiles, backbone_features, molformer_features)
                        
                        # 检查预测结果
                        if torch.isnan(pred).any() or torch.isinf(pred).any():
                            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
                        
                        predictions.extend(pred.cpu().numpy())
                
                # 逆缩放
                predictions = scale_target(np.array(predictions), target_col, inverse=True)
                fold_predictions.append(predictions)
                print(f"✓ Predictions completed for one fold")
                
            except Exception as e:
                print(f"Error processing {model_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 平均预测
        if fold_predictions:
            final_predictions = np.mean(fold_predictions, axis=0)
            
            # 最终清理
            final_predictions = np.nan_to_num(final_predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            submission_df[target_col] = final_predictions
            print(f"✓ Final predictions for {target_col}: mean={np.mean(final_predictions):.6f}, std={np.std(final_predictions):.6f}")
        else:
            submission_df[target_col] = 0.0
    
    # 保存结果
    submission_df.to_csv('submission.csv', index=False)
    print(f'\n✓ Submission saved to submission.csv!')
    print("Sample predictions:")
    print(submission_df.head(10))
    
    # 预测摘要
    print("\n" + "="*70)
    print("                    PREDICTION SUMMARY")
    print("="*70)
    
    print(f"{'Target':<10} {'Available':<10} {'Mean':<15} {'Std':<15} {'Min':<10} {'Max':<10}")
    print("-" * 70)
    
    for target in TARGET_COLUMNS:
        if target in available_targets:
            values = submission_df[target].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            available = "✓"
        else:
            mean_val = std_val = min_val = max_val = 0.0
            available = "✗"
        
        print(f"{target:<10} {available:<10} {mean_val:<15.6f} {std_val:<15.6f} {min_val:<10.6f} {max_val:<10.6f}")
    
    print("-" * 70)
    print(f"Total samples: {len(submission_df)}")
    print(f"Available targets: {len(available_targets)}/{len(TARGET_COLUMNS)}")
    print("="*70)

if __name__ == "__main__":
    main()