# src/modules/self_optimization/reinforcement_learner.py

import os
import logging
import joblib
from typing import Any, Dict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.self_optimization.rl_environment import HermodOptimizationEnv
from src.modules.self_optimization.persistent_memory import PersistentMemory


class ReinforcementLearner:
    """
    Reinforcement Learning Agent for Hermod's Self-Optimization.
    """

    def __init__(self, project_id: str):
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.persistent_memory = PersistentMemory(project_id=project_id)

        self.model_dir = os.path.join(self.config.get('model_dir', f'models/{project_id}/'), 'reinforcement_learning')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'ppo_hermod_rl_agent')

        self.env = HermodOptimizationEnv(project_id=project_id)
        self.env = DummyVecEnv([lambda: self.env])  # Vectorized environment

        self.model = PPO('MlpPolicy', self.env, verbose=1)

    def train(self, total_timesteps: int = 10000):
        """
        Trains the reinforcement learning agent.
        """
        self.logger.info(f"Starting training for {total_timesteps} timesteps.")
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=self.model_dir,
                                                 name_prefix='ppo_hermod_rl_agent')
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
        self.logger.info("Training completed.")

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        self.model.save(self.model_path)
        self.logger.info(f"Model saved at {self.model_path}.")

    def load_model(self):
        """
        Loads a trained model from disk.
        """
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = PPO.load(self.model_path, env=self.env)
            self.logger.info(f"Model loaded from {self.model_path}.")
        else:
            self.logger.warning(f"No model found at {self.model_path}. Starting fresh.")

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """
        Predicts the next action based on the current observation.
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        self.logger.debug(f"Action predicted: {action}")
        return action

    def run_inference(self, steps: int = 100):
        """
        Runs the trained agent in the environment for a specified number of steps.
        """
        obs = self.env.reset()
        for step in range(steps):
            action = self.predict(obs)
            obs, rewards, done, info = self.env.step(action)
            self.env.render()
            if done:
                self.logger.info("Episode finished.")
                break

    def evaluate(self, episodes: int = 10) -> float:
        """
        Evaluates the performance of the trained agent.
        """
        total_rewards = 0.0
        for episode in range(episodes):
            obs = self.env.reset()
            done = False
            episode_rewards = 0.0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_rewards += reward
            total_rewards += episode_rewards
            self.logger.info(f"Episode {episode + 1}: Reward = {episode_rewards}")
        average_reward = total_rewards / episodes
        self.logger.info(f"Average Reward over {episodes} episodes: {average_reward}")
        return average_reward

    def run_sample_operations(self):
        """
        Demonstrates sample operations: training, saving, loading, and inference.
        """
        self.train(total_timesteps=10000)
        self.save_model()
        self.load_model()
        self.evaluate(episodes=5)
        self.run_inference(steps=50)


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize ReinforcementLearner
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    rl_agent = ReinforcementLearner(project_id=project_id)

    # Run sample operations
    rl_agent.run_sample_operations()
