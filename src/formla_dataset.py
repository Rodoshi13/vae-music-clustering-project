"""
formla_dataset.py - 4MuLA Dataset Loader
Multilingual (EN, PT, ES) music with audio, lyrics, and genres
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FourMuLADataset(Dataset):
    """
    4MuLA Dataset Loader
    
    Modes:
    - 'easy': Audio features only
    - 'medium': Audio + Lyrics
    - 'hard': Audio + Lyrics + Genre
    """
    
    def __init__(self, parquet_path, mode='easy', max_lyrics_features=300):
        """
        Args:
            parquet_path (str): Path to 4mula_tiny.parquet
            mode (str): 'easy', 'medium', or 'hard'
            max_lyrics_features (int): TF-IDF dimension
        """
        self.mode = mode
        self.max_lyrics_features = max_lyrics_features
        
        logger.info(f"Loading 4MuLA from {parquet_path}...")
        self.df = pd.read_parquet(parquet_path, engine='pyarrow')

        
        logger.info(f"✓ Loaded {len(self.df)} songs")
        
        # Clean data
        self.df = self.df.dropna(subset=['melspectrogram'])
        if mode in ['medium', 'hard']:
            self.df = self.df.dropna(subset=['music_lyrics'])
        
        logger.info(f"✓ After cleaning: {len(self.df)} songs")
        
        # Process audio
        self._process_audio_features()
        
        # Process lyrics if needed
        if mode in ['medium', 'hard']:
            self._process_lyrics_features()
        
        # Process genres if needed
        if mode == 'hard':
            self._process_genre_labels()
    
    def _process_audio_features(self):
        """Extract melspectrogram features"""
        logger.info("Processing audio features...")
        
        audio_features = []
        valid_indices = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Audio"):
            try:
                melspec = np.array(row['melspectrogram'])
                
                # Mean and std across time
                if len(melspec.shape) == 2:
                    melspec_mean = np.mean(melspec, axis=1)
                    melspec_std = np.std(melspec, axis=1)
                    features = np.concatenate([melspec_mean, melspec_std])
                else:
                    features = melspec.flatten()[:256]
                    if len(features) < 256:
                        features = np.pad(features, (0, 256 - len(features)))
                
                audio_features.append(features)
                valid_indices.append(idx)
            except Exception as e:
                logger.warning(f"Skipping song {idx}: {e}")
        
        self.audio_features = np.array(audio_features, dtype=np.float32)
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        # Normalize
        mean = self.audio_features.mean(axis=0, keepdims=True)
        std = self.audio_features.std(axis=0, keepdims=True) + 1e-8
        self.audio_features = (self.audio_features - mean) / std
        
        logger.info(f"✓ Audio shape: {self.audio_features.shape}")
    
    def _process_lyrics_features(self):
        """Extract TF-IDF features from lyrics"""
        logger.info("Processing lyrics...")
        
        lyrics_texts = self.df['music_lyrics'].fillna('').tolist()
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_lyrics_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words=None
        )
        
        self.lyrics_features = vectorizer.fit_transform(lyrics_texts).toarray()
        self.lyrics_vectorizer = vectorizer
        
        logger.info(f"✓ Lyrics shape: {self.lyrics_features.shape}")
    
    def _process_genre_labels(self):
        """Create genre labels"""
        logger.info("Processing genres...")
        
        unique_genres = self.df['main_genre'].unique()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}
        
        self.genre_labels = np.array([
            self.genre_to_idx[genre] for genre in self.df['main_genre']
        ])
        
        logger.info(f"✓ Found {len(unique_genres)} genres")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        music_id = self.df.iloc[idx]['music_id']
        
        if self.mode == 'easy':
            features = self.audio_features[idx]
            return torch.FloatTensor(features), music_id
        
        elif self.mode == 'medium':
            audio = self.audio_features[idx]
            lyrics = self.lyrics_features[idx]
            features = np.concatenate([audio, lyrics])
            return torch.FloatTensor(features), music_id
        
        elif self.mode == 'hard':
            audio = self.audio_features[idx]
            lyrics = self.lyrics_features[idx]
            features = np.concatenate([audio, lyrics])
            genre = self.genre_labels[idx]
            return torch.FloatTensor(features), torch.LongTensor([genre]), music_id
    
    def get_metadata(self, idx):
        """Get song info"""
        row = self.df.iloc[idx]
        return {
            'music_id': row['music_id'],
            'music_name': row['music_name'],
            'music_lang': row['music_lang'],
            'artist_name': row['art_name'],
            'main_genre': row['main_genre'],
        }


def get_4mula_data_loaders(parquet_path, mode='easy', batch_size=64, test_split=0.2):
    """
    Create train and test loaders
    
    Args:
        parquet_path (str): Path to parquet file
        mode (str): 'easy', 'medium', or 'hard'
        batch_size (int): Batch size
        test_split (float): Test fraction
    
    Returns:
        train_loader, test_loader, dataset
    """
    dataset = FourMuLADataset(parquet_path, mode=mode)
    
    total_size = len(dataset)
    train_size = int(total_size * (1 - test_split))
    test_size = total_size - train_size
    
    logger.info(f"Train: {train_size}, Test: {test_size}")
    
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset
