"""
Dataset utilities for character-level language modeling
"""

import torch
from torch.utils.data import Dataset
import os


class CharDataset(Dataset):
    """
    Character-level dataset for language modeling

    Converts text into sequences of character indices
    """
    def __init__(self, text_file, seq_len=128):
        """
        Args:
            text_file: Path to text file
            seq_len: Length of each training sequence
        """
        self.seq_len = seq_len

        # Read text file
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # Create character vocabulary
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode the entire text
        self.data = [self.char_to_idx[ch] for ch in self.text]

        print(f"Dataset loaded: {len(self.text)} characters, {self.vocab_size} unique")
        print(f"Vocabulary: {''.join(self.chars[:50])}{'...' if len(self.chars) > 50 else ''}")

    def __len__(self):
        # Number of possible sequences
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # Get a sequence of seq_len + 1 characters (input + target)
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode(self, indices):
        """Convert indices to text"""
        return ''.join([self.idx_to_char[i] for i in indices])


class WordDataset(Dataset):
    """
    Word-level dataset for language modeling
    """
    def __init__(self, text_file, seq_len=64):
        """
        Args:
            text_file: Path to text file
            seq_len: Length of each training sequence (in words)
        """
        self.seq_len = seq_len

        # Read text file
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found: {text_file}")

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Simple tokenization (split by whitespace)
        self.words = text.split()

        # Create word vocabulary
        unique_words = sorted(list(set(self.words)))
        self.vocab_size = len(unique_words)

        # Create mappings
        self.word_to_idx = {word: i for i, word in enumerate(unique_words)}
        self.idx_to_word = {i: word for i, word in enumerate(unique_words)}

        # Encode the entire text
        self.data = [self.word_to_idx[word] for word in self.words]

        print(f"Dataset loaded: {len(self.words)} words, {self.vocab_size} unique")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """Convert text to indices"""
        words = text.split()
        return [self.word_to_idx[word] for word in words if word in self.word_to_idx]

    def decode(self, indices):
        """Convert indices to text"""
        return ' '.join([self.idx_to_word[i] for i in indices])
