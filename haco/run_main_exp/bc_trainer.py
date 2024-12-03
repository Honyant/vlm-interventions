import logging
from typing import Dict, Type, Tuple
import numpy as np
import gym
import json
import os
import ray
from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.callbacks import DefaultCallbacks

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

def load_demonstration_data(dir_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load, merge, and split demonstration data from trajectory files."""
    def read_traj(path):
        with open(path, 'r') as f:
            path_data = json.load(f)
        traj = path_data["trajectory"]
        obs = [t[0] for t in traj]
        actions = [t[1] for t in traj]
        return np.array(obs), np.array(actions)

    file_list = os.listdir(dir_path)
    traj_list = []
    for file in file_list:
        if file.endswith('.json'):
            traj_list.append(read_traj(os.path.join(dir_path, file)))
    
    merged_obs = np.concatenate([traj[0] for traj in traj_list], axis=0)
    merged_actions = np.concatenate([traj[1] for traj in traj_list], axis=0)
    
    # print data statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(merged_obs)}")
    print(f"\nAction statistics:")
    print(f"Mean: {np.mean(merged_actions, axis=0)}")
    print(f"Std: {np.std(merged_actions, axis=0)}")
    print(f"Min: {np.min(merged_actions, axis=0)}")
    print(f"Max: {np.max(merged_actions, axis=0)}")
    
    # check for any NaN or infinite values
    print(f"\nData Quality Checks:")
    print(f"NaN in observations: {np.isnan(merged_obs).any()}")
    print(f"Inf in observations: {np.isinf(merged_obs).any()}")
    print(f"NaN in actions: {np.isnan(merged_actions).any()}")
    print(f"Inf in actions: {np.isinf(merged_actions).any()}")
    
    # train/val (90/10 split)
    indices = np.random.permutation(len(merged_obs))
    split_idx = int(0.9 * len(merged_obs))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_data = (merged_obs[train_indices], merged_actions[train_indices])
    val_data = (merged_obs[val_indices], merged_actions[val_indices])
    
    print(f"\nSplit sizes:")
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    return train_data, val_data

class ImprovedBCNetwork(nn.Module):
    """Improved neural network architecture for behavioral cloning."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_layers: int, dropout_rate: float = 0.1):
        super().__init__()
        
        layers = []
        layers.extend([
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        layers.extend([
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        ])
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ImprovedBCPolicy(Policy):
    """Improved behavioral cloning policy with better training stability."""
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        self.network = ImprovedBCNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_hidden_layers"],
            dropout_rate=config.get("dropout_rate", 0.1)
        )
        
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        if config.get("data_dir"):
            (self.train_obs, self.train_actions), (self.val_obs, self.val_actions) = \
                load_demonstration_data(config["data_dir"])
            
            # normalization statistics
            self.obs_mean = np.mean(self.train_obs, axis=0)
            self.obs_std = np.std(self.train_obs, axis=0) + 1e-8
            
            # normalize
            self.train_obs = (self.train_obs - self.obs_mean) / self.obs_std
            self.val_obs = (self.val_obs - self.obs_mean) / self.obs_std
            
            self.batch_size = config["train_batch_size"]
            self.idx = 0
            
            # linear decrease from base to 1-e6
            lambda_fn = lambda step: 1.0 - (step / 10000)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda_fn
            )
    
    def compute_actions(self,
                       obs_batch,
                       state_batches=None,
                       prev_action_batch=None,
                       prev_reward_batch=None,
                       info_batch=None,
                       episodes=None,
                       **kwargs):
        """Compute actions for the current policy."""
        # Normalize observations
        obs_batch = (obs_batch - self.obs_mean) / self.obs_std
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_batch)
            actions = self.network(obs_tensor)
        
        return actions.numpy(), [], {}

    def learn_on_batch(self, samples=None):
        """Train on a batch of demonstrations."""
        # sample batch demonstration data
        if self.idx + self.batch_size > len(self.train_obs):
            perm = np.random.permutation(len(self.train_obs))
            self.train_obs = self.train_obs[perm]
            self.train_actions = self.train_actions[perm]
            self.idx = 0
        
        obs_batch = self.train_obs[self.idx:self.idx + self.batch_size]
        action_batch = self.train_actions[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        
        obs_batch = torch.FloatTensor(obs_batch)
        action_batch = torch.FloatTensor(action_batch)
        
        self.network.train()
        self.optimizer.zero_grad()
        pred_actions = self.network(obs_batch)
        train_loss = nn.MSELoss()(pred_actions, action_batch)
        train_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # validation loss
        self.network.eval()
        with torch.no_grad():
            val_obs = torch.FloatTensor(self.val_obs)
            val_actions = torch.FloatTensor(self.val_actions)
            val_pred = self.network(val_obs)
            val_loss = nn.MSELoss()(val_pred, val_actions)
        
        self.scheduler.step()
        
        return {
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "cur_lr": self.optimizer.param_groups[0]["lr"],
        }

class ImprovedBCTrainer(Trainer):
    """Improved behavioral cloning trainer."""
    _policy_class = ImprovedBCPolicy
    
    _default_config = {
        # === Required Parameters ===
        "env": None,  # Environment class
        "env_config": {},  # Config to pass to env constructor
        
        # === BC-Specific Parameters ===
        "framework": "torch",  # Deep learning framework to use
        "train_batch_size": 256,  # Size of training batches
        "learning_rate": 1e-2,  # Learning rate for optimization
        "hidden_dim": 256,  # Hidden layer dimensions
        "num_hidden_layers": 2,  # Number of hidden layers
        "data_dir": None,  # Directory containing demonstration data
        
        # === RLlib Required Parameters ===
        "num_workers": 0,
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "create_env_on_driver": True,
        
        # === Evaluation Parameters ===
        "evaluation_interval": None,  # Disable evaluation by default
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 0,
        "evaluation_config": {},
        "evaluation_parallel_to_training": False,
        "input_evaluation": [],  # Added this
        
        # === Resource Parameters ===
        "memory_per_worker": 0,
        "object_store_memory_per_worker": 0,
        "custom_resources_per_worker": {},
        
        # === Input/Output Parameters ===
        "input": "sampler",
        "output": "logdir",
        "output_max_file_size": 67108864,
        
        # === Multiagent Parameters ===
        "multiagent": {
            "policies": {},
            "policy_mapping_fn": None,
            "policies_to_train": None,
            "observation_fn": None,
        },
        
        # === Callback and Preprocessing ===
        "callbacks": DefaultCallbacks,
        "preprocessor_pref": "rllib",
        "normalize_actions": False,
        "model": {},
        "optimizer": {},
        
        # === Exploration Parameters ===
        "explore": True,
        "exploration_config": {"type": "StochasticSampling"},
        
        # === Other Parameters ===
        "log_level": "WARN",
        "ignore_worker_failures": False,
        "log_sys_usage": True,
        "fake_sampler": False,
        "seed": None,
        "batch_mode": "truncate_episodes",
        "observation_filter": "NoFilter",
        "input_config": {},
        "policy_states": {},
        "simple_optimizer": True,
        
        # === Framework Specific ===
        "local_tf_session_args": {
            "intra_op_parallelism_threads": 2,
            "inter_op_parallelism_threads": 2,
        },
    }
    
    def __init__(self, config=None, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)
    
    def _init(self, config: TrainerConfigDict, env_creator) -> None:
        """Initialize the trainer.
        
        This is called after __init__ to initialize the trainer with the given config.
        """
        # dummy environment for spaces
        self.env = env_creator(config["env_config"] or {})
        self.policy = ImprovedBCPolicy(
            self.env.observation_space,
            self.env.action_space,
            config
        )
        self.batch_steps = 0
        
    @property
    def _name(self) -> str:
        return "BC"
        
    def step(self) -> Dict:
        results = self.policy.learn_on_batch()
        results.update({
            "timesteps_total": self.batch_steps,
        })
        return results
    
    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "policy.pt")
        state_dict = {
            'network_state': self.policy.network.state_dict(),
            'optimizer_state': self.policy.optimizer.state_dict(),
            'scheduler_state': self.policy.scheduler.state_dict(),
            'obs_mean': self.policy.obs_mean,
            'obs_std': self.policy.obs_std,
        }
        torch.save(state_dict, path)
        return path
    
    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        self.policy.network.load_state_dict(state_dict['network_state'])
        self.policy.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.policy.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.policy.obs_mean = state_dict['obs_mean']
        self.policy.obs_std = state_dict['obs_std']

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing trajectory data")
    parser.add_argument("--num-iters", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Training batch size")
    parser.add_argument("--checkpoint-freq", type=int, default=100,
                       help="Save checkpoint every N iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-2,
                       help="Initial learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of hidden layers")
    args = parser.parse_args()
    
    ray.init()
    
    class DummyEnv(gym.Env):
        def __init__(self, config):
            super().__init__()
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(259,), dtype=np.float32)
            self.action_space = gym.spaces.Box(
                low=-1, high=1, 
                shape=(2,), dtype=np.float32)

        def reset(self):
            return np.zeros(259)

        def step(self, action):
            return np.zeros(259), 0, True, {}
    
    config = {
        "env": DummyEnv,
        "env_config": {},
        "framework": "torch",
        "train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_hidden_layers": args.num_layers,
        "data_dir": args.data_dir,
    }
    
    trainer = ImprovedBCTrainer(config=config)
    
    best_val_loss = float('inf')
    for i in range(args.num_iters):
        result = trainer.train()
        
        metrics = ["train_loss", "val_loss", "cur_lr"]
        metrics_str = ", ".join(f"{k}={result.get(k, 'NaN'):.4f}" for k in metrics)
        if i % 50 == 0:
            print(f"Iteration {i}: {metrics_str}")
        
        # save if validation loss improved
        if result["val_loss"] < best_val_loss:
            best_val_loss = result["val_loss"]
            checkpoint_dir = f"./checkpoints/best"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = trainer.save_checkpoint(checkpoint_dir)
            print(f"New best model saved to {checkpoint_path}")
        
        # regular checkpoint saving
        if i % args.checkpoint_freq == 0:
            checkpoint_dir = f"./checkpoints/iter_{i}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = trainer.save_checkpoint(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_path}")