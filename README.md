<div align="center">
  
# Lightweight Probabilistic Downscaling from a Deterministic Base Model

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.09041-b31b1b.svg)](https://arxiv.org/abs/2609.29383v1)

Code used to train and evaluate climate downscaling models for NeurIPS 2026 workshop paper.
</div>

## Training a model

Everything is run through `run_training.py`. A model is defined by its name, domain, stochasticity type and whether it uses block-sparse attention:

```bash
python run_training.py \
    --model_name my_model \
    --domain ALPS \
    --stochasticity dropout \
    --blocksparse_attention
```

| Argument | Default | Description | Options |
|----------|---------|-------------|---------|
| `--model_name` | required | Name used for the saved model | any name |
| `--domain` | `ALL` | domain for model to be trained on |`ALPS`, `NZ`, `SA`, or `ALL` |
| `--stochasticity` | `dropout` |method of stochasticity used in stage two of training | `dropout`, `perturbations` |
| `--blocksparse_attention` | `False` | Enables block-sparse attention rather than block attention | `True`, `False`|

Trained models are saved to `./models`.

## Evaluating a model

Trained models can be evaluated via opening a notebook and using `evaluate_model(model_name, ...)` provided in `evaluate.py`.
The results will be saved to a csv file and plots for RMSE, P98 bias and PSD will be produced and stored in `./results`.
