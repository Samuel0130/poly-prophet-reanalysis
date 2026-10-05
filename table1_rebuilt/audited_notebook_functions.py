from rdkit import Chem
import pandas as pd

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
