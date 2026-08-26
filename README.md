# Mario DDQN

A Double Deep Q-Network (DDQN) reinforcement learning agent trained to play Super Mario Bros.

## Project Structure

```
agent/
  network.py        # Q-network architecture
  ddqn_agent.py      # DDQN agent (action selection, learning step, target network updates)
  replay_buffer.py   # Experience replay buffer
  exploration.py      # Exploration strategy (e.g. epsilon-greedy)

training/
  trainer.py         # Training loop
  rewards.py          # Reward shaping
  evaluation.py        # Evaluation / testing loop

tests/                # Unit tests
checkpoints/          # Saved model checkpoints (not tracked in git)
logs/                 # Training logs (not tracked in git)
videos/               # Recorded gameplay videos (not tracked in git)

config.yaml           # Training/hyperparameter configuration
main.py                # Entry point
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Tests

```bash
pytest
```
