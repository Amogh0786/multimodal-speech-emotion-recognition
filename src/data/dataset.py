import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Dict
from loguru import logger
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .preprocessor import AudioPreprocessor
from .augmentations import AudioAugmenter

class RAVDESSDataset(Dataset):
    """RAVDESS Dataset mapped to 4 primary emotions."""
    
    # RAVDESS Original: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    # Target: 0: Calm (01, 02), 1: Happy (03, 08), 2: Angry (05, 07), 3: Stressed (04, 06)
    EMOTION_MAP = {
        '01': 0, '02': 0, # Calm
        '03': 1, '08': 1, # Happy
        '05': 2, '07': 2, # Angry
        '04': 3, '06': 3  # Stressed
    }
    
    def __init__(self, data_df: pd.DataFrame, preprocessor: AudioPreprocessor, augmenter: AudioAugmenter = None):
        self.data_df = data_df
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        
    def __len__(self) -> int:
        return len(self.data_df)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.data_df.iloc[idx]
        file_path = row['file_path']
        label = row['label']
        
        waveform = self.preprocessor.process_file(file_path)
        
        if self.augmenter:
            waveform = self.augmenter(waveform)
            
        return waveform, label

def create_folds(raw_dir: str, n_folds: int = 5) -> pd.DataFrame:
    """Parses RAVDESS filenames and creates stratified group k-fold splits."""
    files = list(Path(raw_dir).rglob("*.wav"))
    data = []
    
    for f in files:
        parts = f.stem.split('-')
        if len(parts) != 7:
            continue
        emotion_code = parts[2]
        actor_id = parts[6]
        
        if emotion_code in RAVDESSDataset.EMOTION_MAP:
            label = RAVDESSDataset.EMOTION_MAP[emotion_code]
            data.append({'file_path': str(f), 'label': label, 'actor_id': actor_id})
            
    df = pd.DataFrame(data)
    
    sgkf = StratifiedGroupKFold(n_splits=n_folds)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df, df['label'], groups=df['actor_id'])):
        df.loc[val_idx, 'fold'] = fold
        
    return df
