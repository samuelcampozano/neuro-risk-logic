# Dynamic Model Training System

## Overview
This system enables automatic retraining of the NDD Risk Calculator model using accumulated evaluation data.

## Components

### Scripts
- `train_model.py`: Main training script
- `setup_training.py`: Environment setup

### API Endpoints
- `POST /api/v1/retrain-model`: Trigger model retraining

### Configuration Files
- `training_config.json`: Training parameters
- `auth_config.json`: Authentication tokens
- `current_model_version.json`: Model version tracking

## Usage

### Manual Training
```bash
python scripts/train_model.py
```

### API Training
```bash
curl -X POST "http://localhost:8000/api/v1/retrain-model" \
     -H "Authorization: Bearer your_secret_token_here"
```

### Testing
```bash
python test_training_system.py
```

## Model Versioning
Models are saved as `modelo_entrenado_vX.pkl` in `data/models/` directory.

## Authentication
Update tokens in `auth_config.json` for production use.
