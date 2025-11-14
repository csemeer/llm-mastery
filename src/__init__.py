"""
Small GPT - Educational Language Model Implementation
"""

from .model import GPTModel, count_parameters
from .dataset import CharDataset, WordDataset

__version__ = "0.1.0"
__all__ = ["GPTModel", "count_parameters", "CharDataset", "WordDataset"]
