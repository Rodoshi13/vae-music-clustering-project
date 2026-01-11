"""
formla_dataset_chunked.py - Memory-efficient 4MuLA loader WITH DEBUG
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import logging
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FourMuLADatasetChunked(Dataset):
    """Memory-efficient 4MuLA Dataset"""
    
    def __init__(self, parquet_path, mode='easy', max_samples=200, max_lyrics_features=300):
        self.mode = mode
        self.max_lyrics_features = max_lyrics_features
        
        logger.info(f"Loading first {max_samples} songs from {parquet_path}...")
        
        # Read parquet in batches
        parquet_file = pq.ParquetFile(parquet_path)
        
        batches_to_read = []
        rows_read = 0
        
        for batch in parquet_file.iter_batches(batch_size=50):
            df_batch = batch.to_pandas()
            batches_to_read.append(df_batch)
            rows_read += len(df_batch)
            
            if rows_read >= max_samples:
                break
        
        self.df = pd.concat(batches_to_read, ignore_index=True)
        self.df = self.df.head(max_samples)
        
        logger.info(f"✓ Loaded {len(self.df)} songs")
        logger.info(f"  Columns: {list(self.df.columns)}")
        
        # Clean
        self.df = self.df.dropna(subset=['melspectrogram'])
        if mode in ['medium', 'hard']:
            if 'music_lyrics' in self.df.columns:
                self.df = self.df.dropna(subset=['music_lyrics'])
        
        logger.info(f"✓ After cleaning: {len(self.df)} songs")
        
        # Process
        self._process_audio_features()
        
        if mode in ['medium', 'hard']:
            self._process_lyrics_features()
        
        if mode == 'hard':
            self._process_genre_labels()
    
    def _process_audio_features(self):
        """Extract melspectrogram features - WITH DEBUG"""
        logger.info("Processing audio features...")
        
        audio_features = []
        valid_indices = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Audio"):
            try:
                # Get melspectrogram
                melspec = row['melspectrogram']
                
                # DEBUG first one
                if idx == self.df.index[0]:
                    logger.info(f"DEBUG: melspec type = {type(melspec)}")
                    if hasattr(melspec, 'shape'):
                        logger.info(f"DEBUG: melspec shape = {melspec.shape}")
                
                # Skip if None
                if melspec is None:
                    continue
                
                # Convert to numpy
                if isinstance(melspec, list):
                    melspec = np.array(melspec, dtype=np.float32)
                elif not isinstance(melspec, np.ndarray):
                    try:
                        melspec = np.array(melspec, dtype=np.float32)
                    except:
                        logger.warning(f"Song {idx}: Can't convert to array")
                        continue
                
                # Check if empty
                if melspec.size == 0:
                    continue
                
                # Flatten to 1D
                if len(melspec.shape) > 1:
                    melspec_flat = np.mean(melspec, axis=1)
                else:
                    melspec_flat = melspec.flatten()
                
                # Force to 256
                if len(melspec_flat) >= 256:
                    features = melspec_flat[:256]
                else:
                    features = np.pad(melspec_flat, (0, 256 - len(melspec_flat)))
                
                # Ensure shape
                features = features.reshape(256).astype(np.float32)
                
                audio_features.append(features)
                valid_indices.append(idx)
            
            except Exception as e:
                logger.warning(f"Skipping song {idx}: {str(e)[:100]}")
        
        # CHECK
        if len(audio_features) == 0:
            raise ValueError("NO VALID AUDIO FEATURES! Check your parquet file format.")
        
        logger.info(f"✓ Processed {len(audio_features)} valid songs")
        
        # Stack
        self.audio_features = np.stack(audio_features)
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        
        # Normalize
        mean = self.audio_features.mean(axis=0, keepdims=True)
        std = self.audio_features.std(axis=0, keepdims=True) + 1e-8
        self.audio_features = (self.audio_features - mean) / std
        
        logger.info(f"✓ Audio features shape: {self.audio_features.shape}")
    
    def _process_lyrics_features(self):
        """Extract TF-IDF"""
        logger.info("Processing lyrics...")
        
        if 'music_lyrics' not in self.df.columns:
            logger.warning("No lyrics column found! Skipping...")
            self.lyrics_features = np.zeros((len(self.df), self.max_lyrics_features))
            return
        
        lyrics_texts = self.df['music_lyrics'].fillna('').tolist()
        
        vectorizer = TfidfVectorizer(
            max_features=self.max_lyrics_features,
            min_df=1,
            max_df=0.9,
            ngram_range=(1, 2)
        )
        
        self.lyrics_features = vectorizer.fit_transform(lyrics_texts).toarray()
        logger.info(f"✓ Lyrics features: {self.lyrics_features.shape}")
    
    def _process_genre_labels(self):
        """Create genre labels"""
        logger.info("Processing genres...")
        
        if 'main_genre' not in self.df.columns:
            logger.warning("No genre column! Using dummy genres")
            self.genre_labels = np.zeros(len(self.df), dtype=int)
            self.genre_to_idx = {0: 'unknown'}
            self.idx_to_genre = {0: 'unknown'}
            return
        
        unique_genres = self.df['main_genre'].unique()
        self.genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
        self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}
        
        self.genre_labels = np.array([
            self.genre_to_idx[g] for g in self.df['main_genre']
        ])
        
        logger.info(f"✓ Genres: {len(unique_genres)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        music_id = idx
        
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


def get_4mula_chunked_loaders(parquet_path, mode='easy', max_samples=200, batch_size=8):
    """Create loaders"""
    
    dataset = FourMuLADatasetChunked(parquet_path, mode=mode, max_samples=max_samples)
    
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    test_size = total_size - train_size
    
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset
