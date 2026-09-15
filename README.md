# OCAtari Skiing RL Agent

Reinforcement learning experiments for the Atari `Skiing` environment using OCAtari, Gymnasium, and a DQN agent trained on RAM observations.

The project focuses on how reward shaping changes agent behavior: recurring failure patterns such as collision, missed gates, and stalled movement are translated into reward priorities, then compared through replay generation and score plots.

## Origin

This repository is organized from the PAI Lab, KAIST `seoyeon` branch of [`KAIST-PAI-lab/Atari_DQN`](https://github.com/KAIST-PAI-lab/Atari_DQN/tree/seoyeon). The latest tree is cleaned around the Skiing RL agent work, with local experiment outputs kept out of Git.

## Repository Structure

| Path | Contents |
|---|---|
| `dqn/` | DQN agent and Q-network implementation |
| `training/` | Skiing training variants, including RAM-based DQN and reward-shaping experiments |
| `play/` | Human-play scripts for manually testing the Skiing environment |
| `tools/` | Replay-to-mp4 and score plotting utilities |
| `results/plots/` | Small score-plot images kept as lightweight experiment summaries |
| `legacy/` | Earlier baseline script kept for reference |

Large local artifacts are intentionally excluded: episode history `.pkl` files, model checkpoints, videos, `.npy` datasets, IDE files, and Python caches.

## Main Experiments

- `training/DQN_RAM_skiing.py`: RAM-observation DQN training baseline.
- `training/DQN_skiing_collision.py`: reward shaping around collision, gate miss, and stuck patterns.
- `training/DQN_skiing_pass.py`: reward shaping variant focused on gate-passing behavior.
- `training/DQN_skiing_upgrade.py`: later reward-shaping variant for behavior refinement.
- `tools/replay_to_mp4.py`: converts saved episode histories into replay videos.
- `tools/plot_scores_line.py`, `tools/plot_scores_bar.py`: visualize score progression from local history files.

## Setup

```bash
pip install -r requirements.txt
```

Depending on the local Atari/ALE setup, additional ROM configuration may be required for `ale-py`, `gymnasium`, and `ocatari`.

## Run Examples

Train a RAM-based agent from the repository root:

```bash
python -m training.DQN_RAM_skiing
```

Run a reward-shaping variant:

```bash
python -m training.DQN_skiing_collision
```

Play manually:

```bash
python -m play.human_play_rgb
```

Create a replay video from a local history file:

```bash
python -m tools.replay_to_mp4 --pkl v2/history3206_score-4507.0.pkl --out videos/replay.mp4 --fps 30
```

Plot local score histories:

```bash
python -m tools.plot_scores_line
python -m tools.plot_scores_bar
```

## Local Artifacts Not Tracked

The local `Desktop/Skiing` folder contains trained checkpoints, thousands of episode history files, and replay videos. These are useful for analysis but too noisy for the repository. They are excluded by `.gitignore`; only the code and compact result plots are tracked here.
