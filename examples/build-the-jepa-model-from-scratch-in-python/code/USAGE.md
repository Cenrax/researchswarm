## LeWM JEPA - Usage Guide

### Installation

```bash
pip install -r requirements.txt
```

### Quick Demo

Run the complete demo with synthetic data (no GPU required):

```bash
python demo.py
```

This creates synthetic trajectory data, instantiates the full model, runs training steps, and performs CEM planning.

### Running the Main Script

```bash
python main.py
```

### Running Tests

```bash
python -m pytest tests/ -v
```

### Using as a Library

```python
from lewm.config import LeWMConfig
from lewm.models.lewm import LeWM
from lewm.training.trainer import Trainer
from lewm.planning.cem import CEMPlanner

# Create model
config = LeWMConfig()
model = LeWM(config.encoder, config.predictor)

# Train (requires a DataLoader of trajectory data)
trainer = Trainer(model, config, device="cuda")
trainer.train(dataloader)

# Plan
planner = CEMPlanner(model, config, device="cuda")
actions = planner.plan(obs_current, obs_goal)
next_action = actions[0]
```

### Key Configuration

- `config.encoder.embed_dim = 192` -- latent dimension
- `config.predictor.action_dim = 2` -- set to your environment's action dimension
- `config.sigreg.loss_weight = 0.1` -- SIGReg regularizer weight (lambda)
- `config.training.batch_size = 128` -- batch size for training
- `config.cem.horizon = 5` -- planning horizon

### Architecture

- Encoder: ViT-Tiny (~5.6M params) with BatchNorm1d projection head
- Predictor: 6-layer causal transformer (~3.6M params) with AdaLN action conditioning
- Loss: MSE prediction + 0.1 * SIGReg anti-collapse regularizer
- Planning: Cross-Entropy Method in latent space
