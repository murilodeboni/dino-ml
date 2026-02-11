# Dino ML

A Chrome dinosaur game clone with reinforcement learning agents.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pygame torch numpy matplotlib
```

## Usage

**Play manually:**
```bash
python -m src.manual_play
```
Controls: UP/SPACE to jump, DOWN to duck, ESC to quit.

**Train DQN agent:**
```bash
python -m src.main --mode dqn --fresh    # Train from scratch
python -m src.main --mode dqn            # Resume from checkpoint
python -m src.main --mode dqn --render   # Train with visualization
```

**Evaluate trained model:**
```bash
python -m src.evaluate_dino_agent --model best --episodes 5
python -m src.evaluate_dino_agent --model checkpoint --no-render
```

**Supervised learning:**
```bash
python -m src.main --mode supervised --render
```
