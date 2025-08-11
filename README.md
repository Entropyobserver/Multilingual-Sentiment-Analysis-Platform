# Twitter RoBERTa Fine-tuning: LoRA vs Full Fine-tuning

A comprehensive comparison framework for evaluating LoRA (Low-Rank Adaptation) against full fine-tuning approaches using Twitter RoBERTa for three-class sentiment analysis.

## 🚀 Features

- **Unified Architecture**: Single codebase supporting both LoRA and full fine-tuning
- **Multi-Dataset Training**: Combines IMDB, Stanford Sentiment Treebank, and TweetEval datasets
- **Comprehensive Evaluation**: Statistical analysis with multiple runs and detailed metrics
- **Hyperparameter Optimization**: Optuna-based automatic tuning for both approaches
- **Dual Experiment Tracking**: Integration with MLflow and Weights & Biases
- **Memory Optimization**: GPU memory management and automatic cleanup
- **Extensive Visualization**: Learning curves, confusion matrices, and comparison plots

## 📁 Project Structure

```
twitter_roberta_project/
├── config/                 # Configuration management
│   ├── config.py          # Config class definition
│   └── default_config.yaml # Default parameters
├── data/                   # Data loading and preprocessing
│   ├── loaders.py         # Dataset loading utilities
│   ├── preprocessing.py   # Text cleaning and augmentation
│   └── dataset.py         # PyTorch dataset classes
├── models/                 # Model definitions
│   ├── base_model.py      # Abstract model interface
│   ├── twitter_roberta.py # TwitterRoBERTa wrapper
│   └── adapters.py        # LoRA implementation
├── training/               # Training infrastructure
│   ├── trainer.py         # Training loops and logic
│   ├── evaluator.py       # Metrics and visualization
│   └── utils.py           # Training utilities
├── experiments/            # Experiment orchestration
│   ├── comparison.py      # LoRA vs Full comparison
│   └── optimization.py    # Hyperparameter tuning
├── utils/                  # General utilities
│   ├── logging_utils.py   # Logging configuration
│   └── memory_utils.py    # Memory management
└── main.py                 # Main entry point
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd twitter_roberta_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
export WANDB_API_KEY="your_wandb_api_key"
export HF_TOKEN="your_huggingface_token"  # If using private models
```

## 🚦 Quick Start

### Compare LoRA vs Full Fine-tuning
```bash
python main.py --mode compare
```

### Run LoRA Fine-tuning Only
```bash
python main.py --mode lora
```

### Run Full Fine-tuning Only
```bash
python main.py --mode full
```

### Optimize Hyperparameters
```bash
# Optimize LoRA hyperparameters
python main.py --mode optimize --method lora --trials 20

# Optimize both methods and compare
python main.py --mode optimize --method both --trials 15
```

### Use Custom Configuration
```bash
python main.py --mode compare --config config/custom_config.yaml
```

## ⚙️ Configuration

Create a custom YAML configuration file:

```yaml
# Custom configuration example
model_name: "cardiffnlp/twitter-roberta-base-sentiment-latest"
num_epochs: 4
batch_size: 8
learning_rate: 3e-5
num_runs: 5

# LoRA settings
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1
lora_target_modules: ["query", "value", "key", "dense"]

# Data settings
samples_per_class: 15000
test_samples_per_class: 2000

# Training settings
save_model: true
save_dir: "custom_models"
patience: 3
seed: 42
```

## 📊 Results and Metrics

The framework provides comprehensive evaluation including:

- **Performance Metrics**: Accuracy, F1-score, Precision, Recall
- **Statistical Analysis**: Multiple runs with mean and standard deviation
- **Visualizations**: Learning curves, confusion matrices, error analysis
- **Efficiency Metrics**: Parameter count, training time, memory usage
- **Comparative Analysis**: Side-by-side method comparison

## 🔬 Experiment Tracking

Results are automatically logged to:
- **MLflow**: Local experiment tracking and model registry
- **Weights & Biases**: Cloud-based experiment monitoring and visualization

Access your experiments:
```bash
# View MLflow UI
mlflow ui

# Results are automatically synced to your W&B dashboard
```

## 🎯 Key Features

### LoRA Integration
- Efficient parameter-efficient fine-tuning
- Configurable rank and target modules
- Automatic parameter counting and efficiency metrics

### Multi-Dataset Training
- Balanced sampling from IMDB, SST, and TweetEval
- Automatic data cleaning and preprocessing
- Stratified train/validation splits

### Robust Training
- Early stopping with patience
- Gradient clipping and learning rate scheduling
- Automatic mixed precision support (optional)

### Memory Management
- GPU memory fraction control
- Automatic cache clearing
- Memory usage monitoring and logging

## 🧪 Advanced Usage

### Custom Model Implementation
```python
from models import BaseModel
from config import Config

class CustomRoBERTaModel(BaseModel):
    def __init__(self, config: Config):
        # Implement custom model logic
        pass
```

### Custom Experiment
```python
from experiments import ComparisonExperiment
from config import Config

config = Config(use_lora=True, num_epochs=5)
experiment = ComparisonExperiment(config, device)
results = experiment.run_multiple_runs(config)
```

## 📈 Performance Optimization

### GPU Memory Optimization
- Adjust `batch_size` based on your GPU memory
- Use `torch.cuda.set_per_process_memory_fraction(0.7)` to limit memory usage
- Enable gradient checkpointing for large models (if implemented)

### Training Speed
- Use multiple workers in DataLoader (adjust `num_workers`)
- Enable mixed precision training
- Consider using distributed training for multiple GPUs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Citation

If you use this framework in your research, please cite:

```bibtex
@software{twitter_roberta_lora_comparison,
  title={Twitter RoBERTa Fine-tuning: LoRA vs Full Fine-tuning Comparison},
  author={AI Research Team},
  year={2024},
  url={https://github.com/your-repo/twitter-roberta-project}
}
```

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review example configurations in `config/`

---

**Happy Experimenting! 🎉**