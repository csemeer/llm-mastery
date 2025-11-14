"""
SEC Filings Data Processor

Fetches and processes SEC filings (10-K, 10-Q, 8-K, etc.)
Extracts relevant sections and creates text datasets
"""

import re
import os
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import warnings
from datetime import datetime

try:
    from sec_edgar_downloader import Downloader
except ImportError:
    Downloader = None
    warnings.warn("sec-edgar-downloader not installed. Install with: pip install sec-edgar-downloader")

import torch
from torch.utils.data import Dataset


class SECFilingFetcher:
    """
    Fetch SEC filings for companies
    """
    def __init__(self, company_name: str, email: str, download_dir: str = './data/sec_filings'):
        """
        Args:
            company_name: Your company/name for SEC API
            email: Your email for SEC API
            download_dir: Directory to save filings
        """
        if Downloader is None:
            raise ImportError("sec-edgar-downloader required. Install with: pip install sec-edgar-downloader")

        self.download_dir = download_dir
        self.dl = Downloader(company_name, email, download_dir)

    def fetch_filings(
        self,
        ticker: str,
        filing_type: str = '10-K',
        num_filings: int = 5,
        after_date: Optional[str] = None,
        before_date: Optional[str] = None
    ) -> List[str]:
        """
        Download SEC filings

        Args:
            ticker: Stock ticker
            filing_type: Type of filing ('10-K', '10-Q', '8-K', etc.)
            num_filings: Number of recent filings to download
            after_date: Only filings after this date (YYYY-MM-DD)
            before_date: Only filings before this date (YYYY-MM-DD)

        Returns:
            List of file paths to downloaded filings
        """
        print(f"Fetching {num_filings} {filing_type} filings for {ticker}...")

        self.dl.get(
            filing_type,
            ticker,
            amount=num_filings,
            after=after_date,
            before=before_date
        )

        # Find downloaded files
        filing_dir = os.path.join(self.download_dir, 'sec-edgar-filings', ticker, filing_type)

        if not os.path.exists(filing_dir):
            print(f"No filings found in {filing_dir}")
            return []

        file_paths = []
        for root, dirs, files in os.walk(filing_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.html'):
                    file_paths.append(os.path.join(root, file))

        print(f"Downloaded {len(file_paths)} filings")
        return file_paths


class SECFilingParser:
    """
    Parse and extract relevant information from SEC filings
    """
    # Common section headers in 10-K and 10-Q
    SECTIONS_10K = {
        'item1': r'item\s*1[.\s]*business',
        'item1a': r'item\s*1a[.\s]*risk\s*factors',
        'item7': r'item\s*7[.\s]*management.*discussion',
        'item7a': r'item\s*7a[.\s]*market\s*risk',
        'item8': r'item\s*8[.\s]*financial\s*statements'
    }

    @staticmethod
    def extract_text_from_html(file_path: str) -> str:
        """
        Extract text from HTML SEC filing
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Parse HTML
        soup = BeautifulSoup(content, 'lxml')

        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()

        # Get text
        text = soup.get_text(separator=' ')

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    @staticmethod
    def extract_section(text: str, section_pattern: str, max_chars: int = 50000) -> Optional[str]:
        """
        Extract a specific section from the filing

        Args:
            text: Full filing text
            section_pattern: Regex pattern for section header
            max_chars: Maximum characters to extract

        Returns:
            Section text or None if not found
        """
        # Find section start
        match = re.search(section_pattern, text, re.IGNORECASE)

        if not match:
            return None

        start_idx = match.start()

        # Find next section or end
        # Look for next "Item" heading
        next_section = re.search(r'\n\s*item\s+\d+[a-z]*[.\s]', text[start_idx + 100:], re.IGNORECASE)

        if next_section:
            end_idx = start_idx + 100 + next_section.start()
        else:
            end_idx = start_idx + max_chars

        section_text = text[start_idx:end_idx]

        return section_text

    @staticmethod
    def extract_all_sections(file_path: str) -> Dict[str, str]:
        """
        Extract all major sections from a 10-K filing

        Returns:
            Dictionary mapping section name to text
        """
        text = SECFilingParser.extract_text_from_html(file_path)

        sections = {}

        for section_name, pattern in SECFilingParser.SECTIONS_10K.items():
            section_text = SECFilingParser.extract_section(text, pattern)
            if section_text:
                sections[section_name] = section_text

        return sections

    @staticmethod
    def extract_financial_metrics(text: str) -> Dict[str, str]:
        """
        Extract key financial metrics mentioned in text

        This is a simple pattern-based approach.
        For production, use more sophisticated NER models.
        """
        metrics = {}

        # Revenue patterns
        revenue_pattern = r'revenue[s]?\s*(?:of|was)?\s*\$?\s*([\d,.]+)\s*(million|billion)?'
        revenue_match = re.search(revenue_pattern, text, re.IGNORECASE)
        if revenue_match:
            metrics['revenue'] = revenue_match.group(1)

        # Earnings patterns
        earnings_pattern = r'(?:net\s+income|earnings)\s*(?:of|was)?\s*\$?\s*([\d,.]+)\s*(million|billion)?'
        earnings_match = re.search(earnings_pattern, text, re.IGNORECASE)
        if earnings_match:
            metrics['earnings'] = earnings_match.group(1)

        # EPS patterns
        eps_pattern = r'earnings\s*per\s*share.*?\$?\s*([\d,.]+)'
        eps_match = re.search(eps_pattern, text, re.IGNORECASE)
        if eps_match:
            metrics['eps'] = eps_match.group(1)

        return metrics


class SECFilingDataset(Dataset):
    """
    PyTorch Dataset for SEC filings
    """
    def __init__(
        self,
        file_paths: List[str],
        tokenizer,  # You'd use a proper tokenizer (e.g., from transformers)
        max_length: int = 512,
        sections: Optional[List[str]] = None
    ):
        """
        Args:
            file_paths: List of SEC filing file paths
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
            sections: Specific sections to extract (None = full text)
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sections = sections

        # Parse all filings
        self.samples = []
        for file_path in file_paths:
            try:
                if sections:
                    # Extract specific sections
                    section_texts = SECFilingParser.extract_all_sections(file_path)
                    for section_name, text in section_texts.items():
                        if section_name in sections:
                            self.samples.append({
                                'file': file_path,
                                'section': section_name,
                                'text': text
                            })
                else:
                    # Use full text
                    text = SECFilingParser.extract_text_from_html(file_path)
                    self.samples.append({
                        'file': file_path,
                        'section': 'full',
                        'text': text
                    })
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

        print(f"Loaded {len(self.samples)} samples from {len(file_paths)} filings")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with tokenized text
        """
        sample = self.samples[idx]
        text = sample['text']

        # Tokenize (simplified - use proper tokenizer in practice)
        # This is a placeholder implementation
        tokens = self._simple_tokenize(text)

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'section': sample['section'],
            'file': sample['file']
        }

    def _simple_tokenize(self, text: str) -> List[int]:
        """
        Simple word-level tokenization
        Replace with proper tokenizer (e.g., from transformers)
        """
        # Lowercase and split
        words = text.lower().split()

        # Truncate
        words = words[:self.max_length]

        # Convert to IDs (simplified - just use hash)
        # In practice, use a proper vocabulary
        token_ids = [hash(word) % 10000 for word in words]

        # Pad to max_length
        while len(token_ids) < self.max_length:
            token_ids.append(0)  # Padding token

        return token_ids[:self.max_length]


class MultiModalFinancialDataset(Dataset):
    """
    Combined dataset with both SEC filings and market data
    """
    def __init__(
        self,
        sec_dataset: SECFilingDataset,
        market_dataset,  # MarketDataset from market_data.py
        alignment: str = 'date'  # How to align the two modalities
    ):
        """
        Args:
            sec_dataset: SEC filing dataset
            market_dataset: Market data dataset
            alignment: How to match filings with market data
        """
        self.sec_dataset = sec_dataset
        self.market_dataset = market_dataset
        self.alignment = alignment

        # Create aligned pairs
        # This is simplified - in practice you'd align by filing date
        self.pairs = []

        for i in range(min(len(sec_dataset), len(market_dataset))):
            self.pairs.append((i, i))

        print(f"Created {len(self.pairs)} multi-modal pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with both text and time series data
        """
        sec_idx, market_idx = self.pairs[idx]

        sec_sample = self.sec_dataset[sec_idx]
        market_sample = self.market_dataset[market_idx]

        return {
            **sec_sample,
            **market_sample
        }


if __name__ == "__main__":
    print("Testing SEC Filing Processor...\n")

    # Test HTML parsing
    print("Testing HTML parsing...")

    # Create a sample HTML file for testing
    sample_html = """
    <html>
    <head><title>10-K Filing</title></head>
    <body>
        <h1>Item 1. Business</h1>
        <p>We are a technology company...</p>
        <h1>Item 1A. Risk Factors</h1>
        <p>Our business faces various risks...</p>
    </body>
    </html>
    """

    test_file = '/tmp/test_filing.html'
    with open(test_file, 'w') as f:
        f.write(sample_html)

    text = SECFilingParser.extract_text_from_html(test_file)
    print(f"✓ Extracted {len(text)} characters")

    sections = SECFilingParser.extract_all_sections(test_file)
    print(f"✓ Found {len(sections)} sections: {list(sections.keys())}")

    # Clean up
    os.remove(test_file)

    print("\n✓ Basic tests passed!")
    print("\nNote: To fetch real SEC filings, install: pip install sec-edgar-downloader")
    print("Example usage:")
    print("""
    fetcher = SECFilingFetcher("YourCompany", "your@email.com")
    files = fetcher.fetch_filings("AAPL", "10-K", num_filings=3)
    """)
