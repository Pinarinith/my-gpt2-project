# MiniGPT: A Compact Transformer for Text Generation

MiniGPT is a minimalist implementation of a GPT-like model designed for character-level text generation tasks. This project provides a flexible framework for training, evaluating, and performing inference with a Transformer model, offering various configurations and operating modes.

## ✨ Features

- **Character-level Text Generation**: Trains a Transformer model to generate text character by character
- **Multiple Operating Modes**: Train, evaluate, test, and perform inference with your models
- **Hyperparameter Search**: Built-in functionality to conduct automated hyperparameter tuning
- **Flexible Model Architecture**: Easily adjust layers, attention heads, embedding dimensions, and dropout rates
- **Model Persistence**: Save and load trained models with their configurations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Standard Python libraries (`json`, `os`, `argparse`)

### Installation

Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

The `main.py` script serves as the central entry point for all operations.

### Training and Evaluation

Train a new model from scratch and evaluate its performance:

```bash
python main.py \
    --data_path data/tinyshakespeare.txt \
    --n_layer 4 \
    --n_head 4 \
    --n_embd 256 \
    --dropout 0.1 \
    --learning_rate 0.0003 \
    --epochs 5 \
    --batch_size 64 \
    --model_save_dir models/my_first_model \
    --model_name my_minigpt
```

### Text Generation (Inference)

Generate new text using a pre-trained model:

```bash
python main.py --mode infer \
    --load_model models/my_first_model/best_model.pth \
    --load_config models/my_first_model/best_model_info.json \
    --seed_text "To be or not to be" \
    --max_new_tokens 200 \
    --temperature 0.9 \
    --top_k 50 \
    --num_samples 3
```

### Model Testing

Evaluate a pre-trained model on the test dataset:

```bash
python main.py --mode test \
    --load_model models/my_model/best_model.pth \
    --load_config models/my_model/best_model_info.json \
    --output_file test_results.log
```

### Hyperparameter Search

Conduct automated hyperparameter tuning:

```bash
python main.py --hyperparam_search \
    --config_file config.json \
    --data_path data/tinyshakespeare.txt \
    --model_save_dir models/hyperparam_search_results
```

Example hyperparameter search configuration (`config.json`):

```json
{
    "n_layer": [2, 4, 6],
    "n_head": [2, 4, 8],
    "n_embd": [128, 256, 512],
    "dropout": [0.0, 0.1, 0.2],
    "learning_rate": [0.001, 0.0003, 0.0001]
}
```

## ⚙️ Command-Line Arguments

### Core Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Operation mode: `train_eval`, `infer`, or `test` | `train_eval` |
| `--data_path` | Path to the training data file | `data/tinyshakespeare.txt` |
| `--device` | Device to use: `cuda`, `cpu`, or `auto` | `auto` |
| `--seed` | Random seed for reproducibility | `42` |
| `--verbose` | Print detailed information | `False` |

### Model Architecture

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_layer` | Number of Transformer layers | `4` |
| `--n_head` | Number of attention heads | `4` |
| `--n_embd` | Embedding dimension | `256` |
| `--dropout` | Dropout rate | `0.1` |
| `--block_size` | Context length (sequence length) | `128` |

### Training Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--learning_rate` | Learning rate for the optimizer | `3e-4` |
| `--epochs` | Number of training epochs | `5` |
| `--batch_size` | Batch size for training and evaluation | `64` |
| `--eval_interval` | Evaluation interval in steps | Half of training set size |

### Model Management

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_save_dir` | Directory to save trained models | `models` |
| `--model_name` | Name for the saved model | `minigpt_model` |
| `--load_model` | Path to load a pre-trained model | `None` |
| `--load_config` | Path to load a model configuration file | `None` |
| `--output_file` | Path to save log output | `None` |

### Hyperparameter Search

| Argument | Description | Default |
|----------|-------------|---------|
| `--hyperparam_search` | Enable hyperparameter search | `False` |
| `--config_file` | Path to hyperparameter search config file | `None` |

### Inference Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--seed_text` | Starting text for generation | `ROMEO:` |
| `--max_new_tokens` | Maximum number of new tokens to generate | `500` |
| `--temperature` | Temperature for text generation randomness | `0.8` |
| `--top_k` | Top-k sampling parameter (0 to disable) | `50` |
| `--num_samples` | Number of samples to generate | `1` |

## 📁 Project Structure

```
├── main.py              # Main entry point script
├── train.py             # Training and hyperparameter search functions
├── test.py              # Model testing functionality
├── infer.py             # Text generation/inference
├── model.py             # Define MiniGPT Model
├── dataset.py           # Data loading and preparation
├── config.json          # hyperparmeter search config
├── models/              # Directory for saved models
├── launch/              # train, test or infer help
└── data/               # Training data directory
```

## 🔧 Key Functions

- **`train_single_model`**: Trains a single model configuration
- **`hyperparam_search`**: Conducts automated hyperparameter tuning
- **`run_inference`**: Generates text using a trained model
- **`test_only`**: Evaluates a model on test data
- **`prepare_dataset`**: Handles data loading and character-to-index mappings

## 🎛️ Operating Modes

1. **`train_eval`** (default): Train a new model and evaluate its performance
2. **`infer`**: Generate text using a pre-trained model
3. **`test`**: Evaluate a loaded model on test data

## 🤝 Contributing

Feel free to open issues or pull requests if you have suggestions for improvements or encounter any bugs.

## 📄 License

This project is open source. Please check the repository for license information.