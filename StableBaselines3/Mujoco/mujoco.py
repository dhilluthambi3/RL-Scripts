import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import argparse

def create_environment(env_id, vectorized, n_envs=1):
    """
    Create an environment based on the given parameters.

    Parameters:
        env_id (str): Name of the environment to create.
        vectorized (bool): Whether to create a vectorized environment.
        n_envs (int): The number of environments to create (only applicable if vectorized is True).

    Returns:
        gym.Env or VecEnv: The created environment.
    """
    if vectorized:
        return make_vec_env(env_id, n_envs=n_envs)
    else:
        return gym.make(env_id)

def train_model(algorithm, env, total_timesteps=1000000):
    """
    Train a reinforcement learning model using the specified algorithm.

    Parameters:
        algorithm (str): The algorithm to use for training. Supported values are 'PPO' and 'SAC'.
        env (gym.Env): The environment to train the model on.
        total_timesteps (int): The total number of timesteps to train the model (default: 1000000).

    Returns:
        model: The trained reinforcement learning model.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    
    if algorithm == 'PPO':
        model = sb3.PPO('MlpPolicy', env, verbose=1)
    elif algorithm == 'SAC':
        model = sb3.SAC('MlpPolicy', env, verbose=1)
    else:
        raise ValueError("Unsupported algorithm")

    model.learn(total_timesteps=total_timesteps)
    return model

def save_model(model, filename):
    """
    Save the model to a file.

    Args:
        model (object): The model object to be saved.
        filename (str): The name of the file to save the model to.
    """
    model.save(filename)

def load_model(algorithm, filename, env):
    """
    Load a pre-trained RL model using the specified algorithm.

    Parameters:
        algorithm (str): The RL algorithm to use ('PPO' or 'SAC').
        filename (str): The path to the pre-trained model file.
        env (gym.Env): The Gym environment to use for loading the model.

    Returns:
        model (BaseAlgorithm): The loaded RL model.

    Raises:
        ValueError: If the specified algorithm is not supported.
    """
    if algorithm == 'PPO':
        return sb3.PPO.load(filename, env=env)
    elif algorithm == 'SAC':
        return sb3.SAC.load(filename, env=env)
    else:
        raise ValueError("Unsupported algorithm")

def evaluate_model(model, env_id, n_eval_episodes=10):
    """
    Evaluates the performance of a given RL model on a specified environment.

    Parameters:
        model (object): The RL model to be evaluated.
        env_id (str): Name of the environment to evaluate the model on.
        n_eval_episodes (int): The number of episodes to run during evaluation. Default is 10.

    Returns:
        mean_reward (float): The mean reward obtained by the model during evaluation.
        std_reward (float): The standard deviation of rewards obtained by the model during evaluation.
    """
    eval_env = gym.make(env_id)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    eval_env.close()
    return mean_reward, std_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=['PPO', 'SAC'], required=True)
    parser.add_argument("--vectorized", action='store_true')
    parser.add_argument("--n_envs", type=int, default=1)
    args = parser.parse_args()

    env_id = 'HalfCheetah-v3'
    env = create_environment(env_id, args.vectorized, args.n_envs)

    model = train_model(args.algorithm, env)
    save_model(model, f"{args.algorithm.lower()}_halfcheetah")
    loaded_model = load_model(args.algorithm, f"{args.algorithm.lower()}_halfcheetah", env)

    mean_reward, std_reward = evaluate_model(loaded_model, env_id)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

if __name__ == "__main__":
    main()

# python script.py --algorithm PPO --vectorized --n_envs 4
