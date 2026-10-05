import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
import torch.nn.functional as F

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

# 数据路径
BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'

# 模型参数
SMILES_EMBEDDING_DIM = 256
DROPOUT_RATE = 0.05
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 80
N_FOLDS = 5

# 数据增强配置
DATA_AUGMENTATION = True
AUGMENTATION_PROBABILITY = 0.3

# 目标变量缩放因子
TARGET_SCALING = {
    'FFV': 1000,
    'Tc': 100,
    'Tg': 1,
    'Density': 1,
    'Rg': 1
}

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
# 针对目标的MolFormer特征提取器
# ===============================

class TargetSpecificMolFormerExtractor:
    """针对特定目标属性的MolFormer特征提取器"""
    
    def __init__(self):
        self.model = molformer_model
        self.tokenizer = molformer_tokenizer
        self.is_available = MOLFORMER_AVAILABLE
        self.device = DEVICE
        self.feature_cache = {}
        
        # 目标属性的化学意义和相关特征指导
        self.target_guidance = {
            'Tg': {  # 玻璃化转变温度 - 分子运动性、刚性
                'focus_layers': [6, 7, 8, 9],
                'pooling_strategy': 'max',
                'feature_dim': 32,
                'sequence_focus': 'global'  # 关注全局序列信息
            },
            'FFV': {  # 自由体积分数 - 分子堆积、空隙
                'focus_layers': [4, 5, 6],
                'pooling_strategy': 'mean',
                'feature_dim': 24,
                'sequence_focus': 'local'   # 关注局部结构
            },
            'Tc': {  # 临界温度 - 分子间作用力、极性
                'focus_layers': [7, 8, 9, 10],
                'pooling_strategy': 'attention',
                'feature_dim': 28,
                'sequence_focus': 'interaction'  # 关注相互作用
            },
            'Density': {  # 密度 - 分子量、堆积效率
                'focus_layers': [3, 4, 5],
                'pooling_strategy': 'weighted',
                'feature_dim': 20,
                'sequence_focus': 'structure'    # 关注结构信息
            },
            'Rg': {  # 回转半径 - 分子尺寸、形状
                'focus_layers': [2, 3, 4, 5],
                'pooling_strategy': 'max',
                'feature_dim': 24,
                'sequence_focus': 'shape'        # 关注形状信息
            }
        }
        
        if self.is_available:
            print("✓ Target-specific MolFormer extractor initialized")
            self._setup_projection_layers()
    
    def _setup_projection_layers(self):
        """设置投影层"""
        self.projection_layers = {}
        for target in self.target_guidance:
            dim = self.target_guidance[target]['feature_dim']
            # 创建投影矩阵
            self.projection_layers[target] = torch.randn(768, dim).to(self.device) * 0.1
    
    def get_target_specific_features(self, smiles_list, target):
        """获取针对特定目标的MolFormer特征"""
        if not self.is_available:
            return self._get_fallback_features(smiles_list, target)
        
        try:
            # 批量处理SMILES
            if len(smiles_list) > 32:  # 分批处理大列表
                all_features = []
                for i in range(0, len(smiles_list), 32):
                    batch_smiles = smiles_list[i:i+32]
                    batch_features = self._extract_batch_features(batch_smiles, target)
                    all_features.append(batch_features)
                return np.vstack(all_features)
            else:
                return self._extract_batch_features(smiles_list, target)
                
        except Exception as e:
            print(f"Error in target-specific feature extraction: {e}")
            return self._get_fallback_features(smiles_list, target)
    
    def _extract_batch_features(self, smiles_list, target):
        """批量提取特征"""
        try:
            with torch.no_grad():
                # tokenize
                inputs = self.tokenizer(
                    smiles_list,
                    max_length=256,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 获取隐藏状态
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                # 针对目标处理特征
                target_features = self._process_for_target(hidden_states, target)
                
                return target_features
                
        except Exception as e:
            print(f"Error in batch feature extraction: {e}")
            return self._get_fallback_features(smiles_list, target)
    
    def _process_for_target(self, hidden_states, target):
        """针对特定目标处理隐藏状态"""
        if hidden_states is None:
            return self._get_fallback_features([''] * hidden_states[0].shape[0], target)
        
        guidance = self.target_guidance[target]
        focus_layers = guidance['focus_layers']
        pooling_strategy = guidance['pooling_strategy']
        feature_dim = guidance['feature_dim']
        sequence_focus = guidance['sequence_focus']
        
        # 选择相关层
        selected_layers = []
        for layer_idx in focus_layers:
            if layer_idx < len(hidden_states):
                selected_layers.append(hidden_states[layer_idx])
        
        if not selected_layers:
            return self._get_fallback_features([''] * hidden_states[0].shape[0], target)
        
        # 层级融合策略
        if len(selected_layers) > 1:
            # 根据目标使用不同的层融合策略
            if target in ['Tg', 'Tc']:  # 温度相关，使用注意力加权
                layer_weights = F.softmax(torch.randn(len(selected_layers)).to(self.device), dim=0)
                combined = sum(w * layer for w, layer in zip(layer_weights, selected_layers))
            else:  # 其他目标，使用简单平均
                combined = torch.stack(selected_layers).mean(dim=0)
        else:
            combined = selected_layers[0]
        
        # 序列维度处理 - 根据目标的序列关注点
        if sequence_focus == 'global':
            # 全局信息，使用CLS token + 全局池化
            cls_token = combined[:, 0, :]
            global_pool = combined[:, 1:, :].mean(dim=1)
            pooled = 0.6 * cls_token + 0.4 * global_pool
        elif sequence_focus == 'local':
            # 局部信息，使用最大池化
            pooled = combined.max(dim=1)[0]
        elif sequence_focus == 'interaction':
            # 相互作用，使用注意力池化
            attention_scores = F.softmax(combined.sum(dim=-1), dim=1)
            pooled = (combined * attention_scores.unsqueeze(-1)).sum(dim=1)
        elif sequence_focus == 'structure':
            # 结构信息，加权CLS token
            cls_token = combined[:, 0, :]
            mean_token = combined[:, 1:, :].mean(dim=1)
            pooled = 0.8 * cls_token + 0.2 * mean_token
        elif sequence_focus == 'shape':
            # 形状信息，使用最大和平均的组合
            max_pool = combined.max(dim=1)[0]
            mean_pool = combined.mean(dim=1)
            pooled = torch.cat([max_pool, mean_pool], dim=1)
            # 需要重新调整维度
            pooled = pooled[:, :768]  # 截断到原维度
        else:
            pooled = combined.mean(dim=1)
        
        # 降维到目标维度
        if target in self.projection_layers:
            reduced = torch.matmul(pooled, self.projection_layers[target])
        else:
            # 简单降维
            if feature_dim < pooled.shape[1]:
                indices = torch.linspace(0, pooled.shape[1]-1, feature_dim, dtype=torch.long).to(self.device)
                reduced = pooled[:, indices]
            else:
                padding = torch.zeros(pooled.shape[0], feature_dim - pooled.shape[1]).to(self.device)
                reduced = torch.cat([pooled, padding], dim=1)
        
        return reduced.cpu().numpy()
    
    def _get_fallback_features(self, smiles_list, target):
        """备选特征"""
        guidance = self.target_guidance[target]
        feature_dim = guidance['feature_dim']
        
        features = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    features.append([0.0] * feature_dim)
                    continue
                
                # ... 特征计算代码 ...
                
                # 检查并清理特征值
                clean_features = []
                for f in target_features[:feature_dim] if len(target_features) >= feature_dim else target_features + [0.0] * (feature_dim - len(target_features)):
                    if np.isnan(f) or np.isinf(f):
                        clean_features.append(0.0)
                    else:
                        clean_features.append(float(f))
                
                features.append(clean_features)
                        
            except Exception as e:
                features.append([0.0] * feature_dim)
        
        # 最终检查整个特征数组
        features_array = np.array(features)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features_array

# 全局特征提取器
target_molformer_extractor = TargetSpecificMolFormerExtractor()

# ===============================
# 主链/侧链特征提取 (保持原有逻辑)
# ===============================

def get_backbone_sidechain_features(smiles: str) -> dict:
    """从带有连接点 '*' 的 SMILES 中提取主链和侧链的特征"""
    feature_keys = [
        'bb_atom_count', 'sc_atom_count', 'sc_to_bb_atom_ratio',
        'bb_mw', 'sc_mw', 'sc_to_bb_mw_ratio',
        'bb_rotatable_bonds', 'sc_rotatable_bonds',
        'bb_aromatic_rings', 'sc_aromatic_rings',
        'bb_tpsa', 'sc_tpsa',
        'avg_sc_fragment_size', 'avg_sc_fragment_mw',
        'num_sc_fragments', "stddev_sc_fragment_size", "stddev_sc_fragment_mw"
    ]
    features = {key: np.nan for key in feature_keys}

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
            features['bb_atom_count'] = 1
            features['sc_atom_count'] = 0
            return features

        backbone_path = rdmolops.GetShortestPath(mol, endpoint1, endpoint2)
        if not backbone_path:
            return features
        
        backbone_atom_indices = set(backbone_path)
        all_real_atom_indices = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0}
        sidechain_atom_indices = all_real_atom_indices - backbone_atom_indices

        features['bb_atom_count'] = len(backbone_atom_indices)
        features['sc_atom_count'] = len(sidechain_atom_indices)
        if features['bb_atom_count'] > 0:
            features['sc_to_bb_atom_ratio'] = features['sc_atom_count'] / features['bb_atom_count']
        
        bb_mw = sum(mol.GetAtomWithIdx(i).GetMass() for i in backbone_atom_indices)
        sc_mw = sum(mol.GetAtomWithIdx(i).GetMass() for i in sidechain_atom_indices)

        features['bb_mw'] = bb_mw
        features['sc_mw'] = sc_mw
        if bb_mw > 0:
            features['sc_to_bb_mw_ratio'] = sc_mw / bb_mw

        # 简化其他特征计算以避免错误
        features['bb_rotatable_bonds'] = 0
        features['sc_rotatable_bonds'] = 0
        features['bb_aromatic_rings'] = 0
        features['sc_aromatic_rings'] = 0
        features['bb_tpsa'] = 0
        features['sc_tpsa'] = 0
        
        if sidechain_atom_indices:
            features['num_sc_fragments'] = 1
            features['avg_sc_fragment_size'] = len(sidechain_atom_indices)
            features['avg_sc_fragment_mw'] = sc_mw
            features['stddev_sc_fragment_size'] = 0
            features['stddev_sc_fragment_mw'] = 0
        else:
            features['num_sc_fragments'] = 0
            features['avg_sc_fragment_size'] = 0
            features['avg_sc_fragment_mw'] = 0
            features['stddev_sc_fragment_size'] = 0
            features['stddev_sc_fragment_mw'] = 0
            
    except Exception as e:
        pass
        
    return features

def add_backbone_sidechain_features_to_df(df):
    """将主链和侧链特征计算并添加到DataFrame中"""
    features = df['SMILES'].apply(get_backbone_sidechain_features)
    feature_df = pd.json_normalize(features)
    feature_df = feature_df.fillna(0)
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
# 数据增强
# ===============================

class SMILESAugmenter:
    """SMILES数据增强类"""
    
    def __init__(self, augmentation_prob=0.3):
        self.augmentation_prob = augmentation_prob
        
    def random_smiles(self, smiles):
        """通过随机化原子顺序生成等价SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            
            new_mol = Chem.RenumberAtoms(mol, np.random.permutation(mol.GetNumAtoms()).tolist())
            new_smiles = Chem.MolToSmiles(new_mol, canonical=False)
            
            if Chem.MolFromSmiles(new_smiles) is not None:
                return new_smiles
            else:
                return smiles
        except:
            return smiles
    
    def enumerate_smiles(self, smiles):
        """枚举SMILES的不同表示形式"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            
            augmented_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            
            if Chem.MolFromSmiles(augmented_smiles) is not None:
                return augmented_smiles
            else:
                return smiles
        except:
            return smiles
    
    def augment_smiles(self, smiles):
        """对单个SMILES进行增强"""
        if random.random() > self.augmentation_prob:
            return smiles
        
        augmentation_methods = [
            self.random_smiles,
            self.enumerate_smiles,
        ]
        
        method = random.choice(augmentation_methods)
        return method(smiles)

# ===============================
# 增强的数据集类
# ===============================

class EnhancedPolymerDataset(Dataset):
    def __init__(self, smiles_list, targets, backbone_features, molformer_features, 
                 char_dict, max_len, target_name, is_training=True, augmenter=None):
        self.smiles_list = smiles_list
        self.targets = targets
        self.backbone_features = backbone_features
        self.molformer_features = molformer_features
        self.char_dict = char_dict
        self.max_len = max_len
        self.target_name = target_name
        self.pad_token_id = char_dict['&']
        self.unknown_token_id = char_dict['!']
        self.is_training = is_training
        self.augmenter = augmenter

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        target = self.targets[idx]
        backbone_feats = self.backbone_features[idx]
        molformer_feats = self.molformer_features[idx]
        
        # 只在训练时进行数据增强
        if self.is_training and self.augmenter is not None and DATA_AUGMENTATION:
            original_smiles = smiles
            smiles = self.augmenter.augment_smiles(smiles)
            
            # 如果SMILES被增强了，重新计算MolFormer特征
            if smiles != original_smiles:
                try:
                    new_molformer_feats = target_molformer_extractor.get_target_specific_features([smiles], self.target_name)[0]
                    molformer_feats = new_molformer_feats
                except:
                    pass  # 如果失败，保持原特征
        
        # 对增强后的SMILES进行tokenization
        tokenized_smiles = [self.char_dict.get(char, self.unknown_token_id) for char in smiles]
        padding_len = self.max_len - len(tokenized_smiles)
        
        if padding_len > 0:
            padded_smiles = tokenized_smiles + [self.pad_token_id] * padding_len
        else:
            padded_smiles = tokenized_smiles[:self.max_len]
        
        return (torch.tensor(padded_smiles, dtype=torch.long),
                torch.tensor(backbone_feats, dtype=torch.float),
                torch.tensor(molformer_feats, dtype=torch.float),
                torch.tensor(target, dtype=torch.float))

# ===============================
# 增强的CNN-LSTM模型
# ===============================

class EnhancedAttention(nn.Module):
    """增强的多头注意力机制"""
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super(EnhancedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, lstm_output, mask=None):
        batch_size, seq_len, hidden_size = lstm_output.shape
        
        # 多头注意力
        Q = self.query(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(lstm_output).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # 输出投影
        output = self.output_linear(context)
        
        # 全局池化
        if mask is not None:
            mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1)
            masked_output = output * mask_expanded
            pooled_output = masked_output.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = output.mean(dim=1)
        
        return pooled_output

class TargetSpecificFusion(nn.Module):
    """针对目标的特征融合模块"""
    def __init__(self, target, cnn_dim, lstm_dim, backbone_dim, molformer_dim, output_dim, dropout_rate=0.1):
        super(TargetSpecificFusion, self).__init__()
        self.target = target
        
        # 不同特征的投影层
        self.cnn_proj = nn.Linear(cnn_dim, output_dim // 4)
        self.lstm_proj = nn.Linear(lstm_dim, output_dim // 4)
        self.backbone_proj = nn.Linear(backbone_dim, output_dim // 4)
        self.molformer_proj = nn.Linear(molformer_dim, output_dim // 4)
        
        # 针对目标的门控网络
        gate_input_dim = cnn_dim + lstm_dim + backbone_dim + molformer_dim
        
        if target in ['Tg', 'Tc']:  # 温度相关，更复杂的门控
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(output_dim, 4),  # 4个特征分支的权重
                nn.Softmax(dim=1)
            )
        else:  # 其他目标，简化的门控
            self.gate_network = nn.Sequential(
                nn.Linear(gate_input_dim, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, 4),
                nn.Softmax(dim=1)
            )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, cnn_features, lstm_features, backbone_features, molformer_features):
        # 投影到统一维度
        cnn_proj = self.cnn_proj(cnn_features)
        lstm_proj = self.lstm_proj(lstm_features)
        backbone_proj = self.backbone_proj(backbone_features)
        molformer_proj = self.molformer_proj(molformer_features)
        
        # 计算门控权重
        concat_features = torch.cat([cnn_features, lstm_features, backbone_features, molformer_features], dim=1)
        gate_weights = self.gate_network(concat_features)  # (batch_size, 4)
        
        # 加权融合
        fused_features = (
            gate_weights[:, 0:1] * cnn_proj +
            gate_weights[:, 1:2] * lstm_proj +
            gate_weights[:, 2:3] * backbone_proj +
            gate_weights[:, 3:4] * molformer_proj
        )
        
        fused_features = torch.cat([cnn_proj, lstm_proj, backbone_proj, molformer_proj], dim=1)
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features

class EnhancedPolymerPredictor(nn.Module):
    def __init__(self, target, vocab_size, embedding_dim, hidden_dim, 
                 num_backbone_features, num_molformer_features, dropout_rate=0.3):
        super(EnhancedPolymerPredictor, self).__init__()
        
        self.target = target
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(512, embedding_dim) * 0.1)
        
        # CNN分支 - 针对目标优化
        if target in ['Tg', 'Tc']:  # 温度相关，需要更多尺度的卷积
            cnn_configs = [
                {'out_channels': 64, 'kernel_size': 3},
                {'out_channels': 64, 'kernel_size': 5},
                {'out_channels': 64, 'kernel_size': 7},
                {'out_channels': 32, 'kernel_size': 9},
                {'out_channels': 32, 'kernel_size': 15},
            ]
        else:  # 其他目标，标准配置
            cnn_configs = [
                {'out_channels': 64, 'kernel_size': 3},
                {'out_channels': 64, 'kernel_size': 5},
                {'out_channels': 32, 'kernel_size': 7},
                {'out_channels': 32, 'kernel_size': 11},
            ]
        
        self.cnn_layers = nn.ModuleList()
        for config in cnn_configs:
            self.cnn_layers.append(nn.Conv1d(
                embedding_dim, config['out_channels'], 
                config['kernel_size'], padding='same'
            ))
        
        total_cnn_channels = sum(config['out_channels'] for config in cnn_configs)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        cnn_output_dim = total_cnn_channels * 2  # max + avg pooling
        
        # LSTM分支
        lstm_hidden_size = hidden_dim
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        lstm_output_size = lstm_hidden_size * 2
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # 增强的注意力机制
        self.attention = EnhancedAttention(lstm_output_size, num_heads=4, dropout_rate=dropout_rate)
        
        # 目标特定的特征融合
        fusion_output_dim = 256
        self.feature_fusion = TargetSpecificFusion(
            target=target,
            cnn_dim=cnn_output_dim,
            lstm_dim=lstm_output_size,
            backbone_dim=num_backbone_features,
            molformer_dim=num_molformer_features,
            output_dim=fusion_output_dim,
            dropout_rate=dropout_rate
        )
        
        # 预测头 - 针对目标优化
        if target in ['Tg', 'Tc']:  # 温度相关，更深的网络
            self.predictor = nn.Sequential(
                nn.Linear(fusion_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate // 2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        elif target == 'Density':  # 密度，较直接的关系
            self.predictor = nn.Sequential(
                nn.Linear(fusion_output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:  # FFV, Rg
            self.predictor = nn.Sequential(
                nn.Linear(fusion_output_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def create_padding_mask(self, x, pad_token_id=0):
        """创建padding mask"""
        return (x != pad_token_id).float()
    
    def forward(self, x, backbone_features, molformer_features):
        batch_size, seq_len = x.shape
        
        # 创建padding mask
        padding_mask = self.create_padding_mask(x)
        
        # 嵌入 + 位置编码
        embedded = self.embedding(x)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        embedded = embedded + pos_enc
        embedded = self.embedding_dropout(embedded)
        
        # CNN分支
        cnn_input = embedded.transpose(1, 2)  # (batch, embed_dim, seq_len)
        cnn_features = []
        
        for cnn_layer in self.cnn_layers:
            cnn_out = F.relu(cnn_layer(cnn_input))
            max_pooled = self.global_max_pool(cnn_out).squeeze(-1)
            avg_pooled = self.global_avg_pool(cnn_out).squeeze(-1)
            cnn_features.extend([max_pooled, avg_pooled])
        
        cnn_combined = torch.cat(cnn_features, dim=1)
        
        # LSTM分支
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 增强的注意力机制
        lstm_features = self.attention(lstm_out, padding_mask)
        
        # 特征融合
        fused_features = self.feature_fusion(
            cnn_combined, lstm_features, backbone_features, molformer_features
        )
        
        # 预测
        output = self.predictor(fused_features).squeeze(1)
    
        # 添加数值稳定性检查
        output = torch.clamp(output, min=-1e6, max=1e6)  # 限制输出范围
        
        return output

# ===============================
# 训练和评估函数
# ===============================

def get_model_config(target_col):
    """根据目标变量获取特定的模型配置"""
    if target_col in ['FFV', 'Tc']:
        return {
            'dropout_rate': 0.05,
            'weight_decay': 1e-3,
            'learning_rate': LEARNING_RATE * 0.5,
            'patience': 20
        }
    else:
        return {
            'dropout_rate': DROPOUT_RATE,
            'weight_decay': 1e-4,
            'learning_rate': LEARNING_RATE,
            'patience': 15
        }

def scale_target(values, target_col, inverse=False):
    """缩放或逆缩放目标变量"""
    scale_factor = TARGET_SCALING.get(target_col, 1)
    if inverse:
        return values / scale_factor
    else:
        return values * scale_factor

def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for smiles, backbone_features, molformer_features, targets in train_loader:
        smiles = smiles.to(device)
        backbone_features = backbone_features.to(device)
        molformer_features = molformer_features.to(device)
        targets = targets.to(device)
        
        # 检查输入数据
        if torch.isnan(backbone_features).any():
            backbone_features = torch.nan_to_num(backbone_features, nan=0.0)
        
        if torch.isnan(molformer_features).any():
            molformer_features = torch.nan_to_num(molformer_features, nan=0.0)
        
        if torch.isnan(targets).any():
            targets = torch.nan_to_num(targets, nan=0.0)
        
        optimizer.zero_grad()
        predictions = model(smiles, backbone_features, molformer_features)
        
        # 检查预测值
        if torch.isnan(predictions).any():
            print("Warning: NaN in predictions during training")
            predictions = torch.nan_to_num(predictions, nan=0.0)
        
        loss = criterion(predictions, targets)
        
        # 检查loss
        if torch.isnan(loss):
            print("Warning: NaN loss during training, skipping batch")
            continue
        
        loss.backward()
        
        # 检查梯度
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            print("Warning: NaN gradients detected, skipping batch")
            optimizer.zero_grad()
            continue
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return float('inf')
    
    return total_loss / num_batches

def validate_epoch(model, val_loader, criterion, device, target_col):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for smiles, backbone_features, molformer_features, targets in val_loader:
            smiles = smiles.to(device)
            backbone_features = backbone_features.to(device)
            molformer_features = molformer_features.to(device)
            targets = targets.to(device)
            
            predictions = model(smiles, backbone_features, molformer_features)
            
            # 检查并处理NaN值
            if torch.isnan(predictions).any():
                print(f"Warning: NaN detected in predictions for {target_col}")
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if torch.isnan(targets).any():
                print(f"Warning: NaN detected in targets for {target_col}")
                targets = torch.nan_to_num(targets, nan=0.0, posinf=1e6, neginf=-1e6)
            
            loss = criterion(predictions, targets)
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected for {target_col}")
                continue
            
            total_loss += loss.item()
            num_batches += 1
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    if num_batches == 0:
        return float('inf'), float('inf')
    
    # 转换为numpy数组并处理NaN
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 处理预测中的NaN和无穷值
    all_predictions = np.nan_to_num(all_predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    all_targets = np.nan_to_num(all_targets, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 计算原始尺度的MAE
    original_predictions = scale_target(all_predictions, target_col, inverse=True)
    original_targets = scale_target(all_targets, target_col, inverse=True)
    
    # 再次检查和处理NaN
    original_predictions = np.nan_to_num(original_predictions, nan=0.0, posinf=1e6, neginf=-1e6)
    original_targets = np.nan_to_num(original_targets, nan=0.0, posinf=1e6, neginf=-1e6)
    
    try:
        mae = mean_absolute_error(original_targets, original_predictions)
        if np.isnan(mae) or np.isinf(mae):
            mae = float('inf')
    except Exception as e:
        print(f"Error calculating MAE for {target_col}: {e}")
        mae = float('inf')
    
    return total_loss / num_batches, mae

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                epochs, device, target_col, patience=15):
    """主训练循环"""
    model.to(device)
    best_val_loss = np.inf
    best_model_state = None
    best_mae = np.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device, target_col)
        
        # 学习率调度
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        
        # 早停和最佳模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  -> Better model found! Val Loss: {best_val_loss:.6f}, Val MAE: {val_mae:.6f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with Val Loss: {best_val_loss:.6f}, Final MAE: {best_mae:.6f}")
    
    return model, best_mae

def add_extra_data(df_train, df_extra, target):
    """合并外部数据"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    df_extra = df_extra.dropna(subset=['SMILES', target])
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])
    
    for smile in cross_smiles:
        if pd.isnull(df_train.loc[df_train['SMILES']==smile, target]).any():
            impute_value = df_extra[df_extra['SMILES']==smile][target].values[0]
            df_train.loc[df_train['SMILES']==smile, target] = impute_value
            
    df_train = pd.concat([df_train, df_extra[df_extra['SMILES'].isin(unique_smiles_extra)]], 
                        axis=0, ignore_index=True)
    
    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'For target "{target}": {n_samples_after - n_samples_before} new samples were added.')
    return df_train

# ===============================
# 主函数
# ===============================

def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")
    print(f"MolFormer available: {MOLFORMER_AVAILABLE}")
    print(f"Data Augmentation: {'Enabled' if DATA_AUGMENTATION else 'Disabled'}")
    
    # 加载数据
    try:
        train_df = pd.read_csv(BASE_PATH + 'train.csv')
        test_df = pd.read_csv(BASE_PATH + 'test.csv')
    except FileNotFoundError:
        print(f"Error: Data files not found at {BASE_PATH}")
        return
    
    # 清洗SMILES
    train_df['SMILES'] = train_df['SMILES'].apply(clean_and_validate_smiles)
    test_df['SMILES'] = test_df['SMILES'].apply(clean_and_validate_smiles)
    
    train_df = train_df[train_df['SMILES'].notnull()].reset_index(drop=True)
    test_df = test_df[test_df['SMILES'].notnull()].reset_index(drop=True)
    
    # 添加主链/侧链特征
    print("Calculating backbone/sidechain features...")
    train_df = add_backbone_sidechain_features_to_df(train_df)
    test_df = add_backbone_sidechain_features_to_df(test_df)
    
    # 加载额外数据
    try:
        extra_paths = {
            'tc': BASE_PATH + 'train_supplement/dataset1.csv',
            'tg': BASE_PATH + 'train_supplement/dataset3.csv',
            'ffv': BASE_PATH + 'train_supplement/dataset4.csv'
        }
        
        data_tc = pd.read_csv(extra_paths['tc']).rename(columns={'TC_mean': 'Tc'})
        train_df = add_extra_data(train_df, data_tc, 'Tc')
        
        data_tg = pd.read_csv(extra_paths['tg'], usecols=['SMILES', 'Tg'])
        train_df = add_extra_data(train_df, data_tg, 'Tg')
        
        data_ffv = pd.read_csv(extra_paths['ffv'], usecols=['SMILES', 'FFV'])
        train_df = add_extra_data(train_df, data_ffv, 'FFV')
        
    except Exception as e:
        print(f"Warning: Could not load extra data: {e}")
    
    # 创建词汇表
    unique_smiles_chars = sorted(list(set(''.join(train_df['SMILES'].astype(str))) - set(['&', '!'])))
    smiles_char_dict = {char: i + 1 for i, char in enumerate(unique_smiles_chars)}
    smiles_char_dict['&'] = 0  # 填充token
    smiles_char_dict['!'] = len(smiles_char_dict)  # 未知token
    
    vocab_size = len(smiles_char_dict)
    max_smiles_len = train_df['SMILES'].astype(str).str.len().max() + 20
    
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max SMILES Length: {max_smiles_len}")
    
    # 创建数据增强器
    augmenter = SMILESAugmenter(augmentation_prob=AUGMENTATION_PROBABILITY)
    
    # 全局训练结果
    TRAINING_RESULTS = {}
    
    # 为每个目标训练模型
    for target_col in TARGET_COLUMNS:
        print(f"\n{'='*50}")
        print(f"Training Enhanced CNN-LSTM for: {target_col}")
        print(f"{'='*50}")
        
        df_filtered = train_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
        
        if len(df_filtered) < 20:
            print(f"Not enough data for '{target_col}'. Skipping...")
            continue
        
        # 数据统计
        print(f"{target_col} statistics:")
        print(f"Range: {df_filtered[target_col].min():.6f} - {df_filtered[target_col].max():.6f}")
        print(f"Mean: {df_filtered[target_col].mean():.6f}, Std: {df_filtered[target_col].std():.6f}")
        print(f"Unique values: {df_filtered[target_col].nunique()}")
        
        # 缩放目标变量
        df_filtered[target_col] = scale_target(df_filtered[target_col].values, target_col)
        
        # 获取MolFormer特征
        print(f"Extracting MolFormer features for {target_col}...")
        molformer_features = target_molformer_extractor.get_target_specific_features(
            df_filtered['SMILES'].tolist(), target_col
        )
        
        # 获取配置
        model_config = get_model_config(target_col)
        print(f"Using config: LR={model_config['learning_rate']:.6f}, "
              f"Dropout={model_config['dropout_rate']:.3f}, "
              f"Patience={model_config['patience']}")
        
        # K折交叉验证
        try:
            y_bins = pd.qcut(df_filtered[target_col], q=10, labels=False, duplicates='drop')
            stratify_param = y_bins
        except ValueError:
            stratify_param = None
        
        kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_models = []
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df_filtered['SMILES'], stratify_param)):
            print(f"\nTraining Fold {fold+1}/{N_FOLDS}")
            
            X_train_smiles = df_filtered.iloc[train_idx]['SMILES'].values
            X_val_smiles = df_filtered.iloc[val_idx]['SMILES'].values
            y_train = df_filtered.iloc[train_idx][target_col].values
            y_val = df_filtered.iloc[val_idx][target_col].values
            X_train_backbone = df_filtered.iloc[train_idx][BACKBONE_SIDECHAIN_FEATURE_NAMES].values
            X_val_backbone = df_filtered.iloc[val_idx][BACKBONE_SIDECHAIN_FEATURE_NAMES].values
            X_train_molformer = molformer_features[train_idx]
            X_val_molformer = molformer_features[val_idx]
            
            # 创建数据集
            train_dataset = EnhancedPolymerDataset(
                X_train_smiles, y_train, X_train_backbone, X_train_molformer,
                smiles_char_dict, max_smiles_len, target_col,
                is_training=True, augmenter=augmenter
            )
            val_dataset = EnhancedPolymerDataset(
                X_val_smiles, y_val, X_val_backbone, X_val_molformer,
                smiles_char_dict, max_smiles_len, target_col,
                is_training=False, augmenter=None
            )
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # 创建模型
            num_molformer_features = target_molformer_extractor.target_guidance[target_col]['feature_dim']
            model = EnhancedPolymerPredictor(
                target=target_col,
                vocab_size=vocab_size,
                embedding_dim=SMILES_EMBEDDING_DIM,
                hidden_dim=128,
                num_backbone_features=len(BACKBONE_SIDECHAIN_FEATURE_NAMES),
                num_molformer_features=num_molformer_features,
                dropout_rate=model_config['dropout_rate']
            )
            
            # 优化器和调度器
            optimizer = optim.AdamW(model.parameters(), 
                                  lr=model_config['learning_rate'], 
                                  weight_decay=model_config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, verbose=False
            )
            criterion = nn.MSELoss()
            
            # 训练
            model, fold_mae = train_model(model, train_loader, val_loader, optimizer, scheduler, 
                                        criterion, EPOCHS, DEVICE, target_col, patience=model_config['patience'])
            
            fold_models.append(model.state_dict().copy())
            fold_maes.append(fold_mae)
        
        # 记录结果
        avg_mae = np.mean(fold_maes)
        TRAINING_RESULTS[target_col] = {
            'avg_mae': avg_mae,
            'fold_maes': fold_maes,
            'data_samples': len(df_filtered),
            'models': fold_models
        }
        print(f"\n{target_col} K-Fold Average MAE: {avg_mae:.6f}")
    
    # 预测测试数据
    print("\n" + "="*50)
    print("Predicting test data...")
    print("="*50)
    
    submission_df = pd.DataFrame({'id': test_df['id']})
    
    for target_col in TARGET_COLUMNS:
        if target_col not in TRAINING_RESULTS:
            submission_df[target_col] = 0.0
            continue
        
        print(f"\nPredicting for: {target_col}")
        
        # 获取测试集的MolFormer特征
        test_molformer_features = target_molformer_extractor.get_target_specific_features(
            test_df['SMILES'].tolist(), target_col
        )
        
        fold_predictions = []
        model_config = get_model_config(target_col)
        
        for fold, model_state in enumerate(TRAINING_RESULTS[target_col]['models']):
            # 重建模型
            num_molformer_features = target_molformer_extractor.target_guidance[target_col]['feature_dim']
            model = EnhancedPolymerPredictor(
                target=target_col,
                vocab_size=vocab_size,
                embedding_dim=SMILES_EMBEDDING_DIM,
                hidden_dim=128,
                num_backbone_features=len(BACKBONE_SIDECHAIN_FEATURE_NAMES),
                num_molformer_features=num_molformer_features,
                dropout_rate=model_config['dropout_rate']
            )
            model.load_state_dict(model_state)
            model.to(DEVICE)
            model.eval()
            
            # 创建测试数据集
            test_dataset = EnhancedPolymerDataset(
                test_df['SMILES'].values, 
                np.zeros(len(test_df)),  # dummy targets
                test_df[BACKBONE_SIDECHAIN_FEATURE_NAMES].values,
                test_molformer_features,
                smiles_char_dict, max_smiles_len, target_col,
                is_training=False, augmenter=None
            )
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            
            # 预测
            predictions = []
            with torch.no_grad():
                for smiles, backbone_features, molformer_features, _ in tqdm(test_loader, desc=f"Fold {fold+1}"):
                    smiles = smiles.to(DEVICE)
                    backbone_features = backbone_features.to(DEVICE)
                    molformer_features = molformer_features.to(DEVICE)
                    
                    pred = model(smiles, backbone_features, molformer_features)
                    predictions.extend(pred.cpu().numpy())
            
            # 逆缩放
            predictions = scale_target(np.array(predictions), target_col, inverse=True)
            fold_predictions.append(predictions)
        
        # 平均预测
        final_predictions = np.mean(fold_predictions, axis=0)
        submission_df[target_col] = final_predictions
    
    # 保存结果
    submission_df.to_csv('submission_enhanced_cnnlstm_molformer.csv', index=False)
    print(f'\nSubmission saved!')
    print("Sample predictions:")
    print(submission_df.head())
    
    # 打印训练摘要
    print("\n" + "="*70)
    print("                    TRAINING SUMMARY")
    print("="*70)
    
    total_mae = 0
    valid_targets = 0
    
    print(f"{'Target':<10} {'Data Samples':<12} {'Validation MAE':<15} {'Details':<25}")
    print("-" * 70)
    
    for target, results in TRAINING_RESULTS.items():
        data_samples = results['data_samples']
        avg_mae = results['avg_mae']
        fold_maes = results['fold_maes']
        mae_std = np.std(fold_maes)
        details = f"±{mae_std:.4f} (K-Fold)"
        total_mae += avg_mae
        valid_targets += 1
        print(f"{target:<10} {data_samples:<12} {avg_mae:<15.6f} {details:<25}")
    
    print("-" * 70)
    
    if valid_targets > 0:
        overall_avg_mae = total_mae / valid_targets
        print(f"{'OVERALL':<10} {'':<12} {overall_avg_mae:<15.6f} {'Average MAE':<25}")
    
    print(f"\nModel Configuration:")
    print(f"- Architecture: Enhanced CNN-LSTM with MolFormer features")
    print(f"- MolFormer Available: {MOLFORMER_AVAILABLE}")
    print(f"- Target-specific feature dimensions: {[target_molformer_extractor.target_guidance[t]['feature_dim'] for t in TARGET_COLUMNS]}")
    print(f"- Data Augmentation: {'Enabled' if DATA_AUGMENTATION else 'Disabled'}")
    print(f"- Device: {DEVICE}")
    print("="*70)

if __name__ == "__main__":
    main()