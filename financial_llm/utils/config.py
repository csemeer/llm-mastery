"""
Configuration Management for Financial LLM

Handles loading and validation of configuration files
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import warnings


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 50000
    d_model: int = 256
    num_heads: int = 8
    num_text_layers: int = 4
    num_ts_layers: int = 3
    num_fusion_layers: int = 2
    d_ff: int = 1024
    ts_input_dim: int = 7
    num_indicators: int = 20
    max_seq_len_text: int = 512
    max_seq_len_ts: int = 60
    dropout: float = 0.1
    num_trading_actions: int = 3
    enable_risk_prediction: bool = True
    enable_portfolio_optimization: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    seq_len: int = 60
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    scheduler: str = "onecycle"
    warmup_epochs: int = 5
    num_workers: int = 4
    pin_memory: bool = True
    checkpoint_dir: str = "checkpoints/financial_llm"
    save_every: int = 10
    keep_best: bool = True


@dataclass
class ContinualLearningConfig:
    """Continual learning configuration"""
    enabled: bool = True
    ewc_lambda: float = 100.0
    fisher_samples: int = 100
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.3
    lora_enabled: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_target_modules: list = field(default_factory=lambda: ["W_q", "W_v"])
    lora_dropout: float = 0.1


@dataclass
class DataConfig:
    """Data configuration"""
    tickers: list = field(default_factory=lambda: ["AAPL"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    interval: str = "1d"
    indicators_enabled: bool = True
    indicator_types: list = field(default_factory=lambda: ["sma", "ema", "rsi", "macd"])
    normalize: bool = True
    fill_na_method: str = "ffill"


@dataclass
class FinancialLLMConfig:
    """Complete configuration for Financial LLM"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    continual_learning: ContinualLearningConfig = field(default_factory=ContinualLearningConfig)
    data: DataConfig = field(default_factory=DataConfig)


class ConfigLoader:
    """
    Load and manage configurations
    """

    @staticmethod
    def load_from_yaml(config_path: str) -> FinancialLLMConfig:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to YAML configuration file

        Returns:
            FinancialLLMConfig object
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return ConfigLoader._dict_to_config(config_dict)

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> FinancialLLMConfig:
        """Convert dictionary to FinancialLLMConfig"""

        # Model config
        model_dict = config_dict.get('model', {})
        model_config = ModelConfig(**model_dict)

        # Training config
        training_dict = config_dict.get('training', {})
        training_config = TrainingConfig(**training_dict)

        # Continual learning config
        cl_dict = config_dict.get('continual_learning', {})
        cl_config = ContinualLearningConfig(
            enabled=cl_dict.get('enabled', True),
            ewc_lambda=cl_dict.get('ewc', {}).get('lambda', 100.0),
            fisher_samples=cl_dict.get('ewc', {}).get('fisher_samples', 100),
            replay_buffer_size=cl_dict.get('replay', {}).get('buffer_size', 10000),
            replay_ratio=cl_dict.get('replay', {}).get('replay_ratio', 0.3),
            lora_enabled=cl_dict.get('lora', {}).get('enabled', False),
            lora_rank=cl_dict.get('lora', {}).get('rank', 8),
            lora_alpha=cl_dict.get('lora', {}).get('alpha', 16.0),
            lora_target_modules=cl_dict.get('lora', {}).get('target_modules', ["W_q", "W_v"]),
            lora_dropout=cl_dict.get('lora', {}).get('dropout', 0.1)
        )

        # Data config
        data_dict = config_dict.get('data', {})
        market_dict = data_dict.get('market', {})
        indicators_dict = data_dict.get('indicators', {})
        preprocessing_dict = data_dict.get('preprocessing', {})

        data_config = DataConfig(
            tickers=market_dict.get('tickers', ["AAPL"]),
            start_date=market_dict.get('start_date', "2020-01-01"),
            end_date=market_dict.get('end_date', "2024-01-01"),
            interval=market_dict.get('interval', "1d"),
            indicators_enabled=indicators_dict.get('enabled', True),
            indicator_types=indicators_dict.get('types', ["sma", "ema", "rsi"]),
            normalize=preprocessing_dict.get('normalize', True),
            fill_na_method=preprocessing_dict.get('fill_na_method', "ffill")
        )

        return FinancialLLMConfig(
            model=model_config,
            training=training_config,
            continual_learning=cl_config,
            data=data_config
        )

    @staticmethod
    def load_preset(preset: str = "default") -> FinancialLLMConfig:
        """
        Load a preset configuration

        Args:
            preset: "small", "default", or "large"

        Returns:
            FinancialLLMConfig object
        """
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'configs'
        )

        preset_files = {
            'small': 'small_model.yaml',
            'default': 'model_config.yaml',
            'large': 'large_model.yaml'
        }

        if preset not in preset_files:
            warnings.warn(f"Unknown preset '{preset}', using 'default'")
            preset = 'default'

        config_path = os.path.join(config_dir, preset_files[preset])

        if not os.path.exists(config_path):
            warnings.warn(f"Preset file not found: {config_path}. Using default values.")
            return FinancialLLMConfig()

        return ConfigLoader.load_from_yaml(config_path)

    @staticmethod
    def save_to_yaml(config: FinancialLLMConfig, output_path: str):
        """
        Save configuration to YAML file

        Args:
            config: FinancialLLMConfig object
            output_path: Path to save YAML file
        """
        config_dict = {
            'model': config.model.__dict__,
            'training': config.training.__dict__,
            'continual_learning': config.continual_learning.__dict__,
            'data': config.data.__dict__
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {output_path}")


if __name__ == "__main__":
    print("Testing configuration management...\n")

    # Test loading preset
    print("Loading 'small' preset...")
    config = ConfigLoader.load_preset('small')

    print(f"✓ Model d_model: {config.model.d_model}")
    print(f"✓ Training batch_size: {config.training.batch_size}")
    print(f"✓ CL enabled: {config.continual_learning.enabled}")
    print(f"✓ Data tickers: {config.data.tickers}")

    # Test saving
    print("\nSaving configuration...")
    output_path = "/tmp/test_config.yaml"
    ConfigLoader.save_to_yaml(config, output_path)

    # Test reloading
    print("\nReloading configuration...")
    reloaded_config = ConfigLoader.load_from_yaml(output_path)
    print(f"✓ Reloaded d_model: {reloaded_config.model.d_model}")

    # Clean up
    os.remove(output_path)

    print("\n✓ Configuration management working!")
