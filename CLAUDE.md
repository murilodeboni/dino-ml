# Claude Code Context

## Project Overview

Chrome dinosaur game clone with Deep Q-Network (DQN) reinforcement learning. The agent learns to jump over trees and duck under birds.

## Key Files

- `src/dino_env.py` - Gym-like environment with frame skipping (4 frames per action)
- `src/train_dqn.py` - Double DQN training loop with replay buffer
- `src/model.py` - Neural network (64 hidden units, 12 inputs, 3 outputs)
- `src/Dino.py` - Dino physics (jump, duck, gravity)
- `src/Obstacle.py` - Trees and birds
- `src/constants.py` - Game constants (screen size, physics)

## Architecture Decisions

**State (12 features, all normalized 0-1):**
- Distance to next obstacle, height, type (tree=0, bird=1), y position
- Dino y position, is_jumping, is_ducking, velocity_y
- Distance to second obstacle, obstacle speed
- Frames since jump, frames since duck

**Actions:** 0=run/unduck, 1=jump, 2=duck

**Rewards:** +1 survive, +20 pass obstacle, -20 die

**Key hyperparameters:**
- Frame skip: 4 (each action repeats for 4 frames)
- Replay buffer: 20,000
- Epsilon decay: 0.995
- Learning rate: 5e-4
- Hidden layer: 64 units

## Commands

```bash
python -m src.main --mode dqn --fresh   # Train from scratch
python -m src.main --mode dqn           # Resume training
python -m src.evaluate_dino_agent       # Evaluate best model
python -m src.manual_play               # Play manually
```
