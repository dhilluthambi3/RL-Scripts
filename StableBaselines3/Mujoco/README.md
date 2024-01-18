
# MuJoCo Agent Training with Stable Baselines 3

This repository contains a Python script for training reinforcement learning agents on MuJoCo environments using the Stable Baselines 3 library. The script supports both Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) algorithms and allows for easy configuration of training parameters via command-line arguments.

## Features

- **Flexible Agent Training**: Train agents using either PPO or SAC algorithms.
- **Vectorized Environments**: Option to use vectorized environments for faster training.
- **Command-Line Interface**: Easily configure training parameters through command-line arguments.

## Requirements

To run this script, you need Python 3.6+ and the following packages:
- `gymnasium`
- `stable-baselines3`

You can install these packages using pip:
```
pip install gymnasium stable-baselines3
```

## Usage

To use this script, clone the repository and run the script from the command line. Here is a basic example:

```
python your_script.py --algorithm PPO --vectorized --n_envs 4
```

### Command-Line Arguments

- `--algorithm`: The algorithm to use for training (`PPO` or `SAC`). This argument is required.
- `--vectorized`: If included, uses a vectorized environment.
- `--n_envs`: The number of environments to use for vectorization. Default is 1. Only relevant if `--vectorized` is set.

## Example

Train a PPO agent with 4 parallel environments:

```
python mujoco.py --algorithm PPO --vectorized --n_envs 4
```

Train a SAC agent without vectorization:

```
python mujoco.py --algorithm SAC
```

## License

This project is open source and available under the [MIT License](LICENSE).
