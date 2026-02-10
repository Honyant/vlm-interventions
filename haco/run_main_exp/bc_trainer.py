import logging
from typing import Dict, Tuple, Sequence, Union
import numpy as np
import gym
import json
import os
import ray
from pathlib import Path
from ray.rllib.agents import Trainer
from ray.rllib.utils.typing import TrainerConfigDict
from ray.rllib.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.callbacks import DefaultCallbacks

torch, nn = try_import_torch()
logger = logging.getLogger(__name__)


class DummyEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

    def reset(self):
        return np.zeros(259, dtype=np.float32)

    def step(self, action):
        return np.zeros(259, dtype=np.float32), 0.0, True, {}


# ----------------------------
# Data loader (preserves adjacency)
# ----------------------------
def load_demonstration_data(data_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load, merge, and split demonstration data from trajectory files, preserving adjacency.

    Supports VLM metadata in trajectory entries:
    - If entry has 5+ elements and element[4] is a float in [-1, 1]: treated as weight
    - If entry has 5+ elements and element[4] is a bool: treated as filtering flag
    """
    def read_traj(path: Path):
        with path.open('r') as f:
            path_data = json.load(f)
        traj = path_data["trajectory"]  # list of [obs, act, intervention, image, metadata?]

        obs = []
        actions = []
        weights = []

        for t in traj:
            # Check for filtering flag (5th element is boolean)
            if len(t) >= 5 and isinstance(t[4], bool):
                if not t[4]:  # include=False, skip this sample
                    continue

            obs.append(t[0])
            actions.append(t[1])

            # Check for weight (5th element is float)
            if len(t) >= 5 and isinstance(t[4], (int, float)) and not isinstance(t[4], bool):
                weights.append(float(t[4]))
            else:
                weights.append(1.0)  # default weight

        if not obs:  # All samples filtered out
            return None, None, None

        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(weights, dtype=np.float32),
        )

    if isinstance(data_source, (str, os.PathLike)):
        dir_path = Path(data_source)
        if not dir_path.exists():
            raise ValueError(f"Data directory not found: {dir_path}")
        file_paths = sorted(dir_path.glob("*.json"))
    else:
        file_paths = sorted(Path(p) for p in data_source)
    if not file_paths:
        raise ValueError(f"No .json trajectories found in {data_source}")

    obs_list, act_list, prev_obs_list, prev_act_list, has_prev_list, weight_list = [], [], [], [], [], []

    for file_path in file_paths:
        result = read_traj(file_path)
        if result[0] is None:  # All samples filtered out
            continue
        o, a, w = result
        n = len(o)
        if n == 0:
            continue
        # per-trajectory previous pointers (duplicate first to keep shape)
        prev_o = np.vstack([o[0:1], o[:-1]])
        prev_a = np.vstack([a[0:1], a[:-1]])
        has_prev = np.zeros((n, 1), dtype=np.float32)
        has_prev[1:, 0] = 1.0

        obs_list.append(o); act_list.append(a); weight_list.append(w)
        prev_obs_list.append(prev_o); prev_act_list.append(prev_a)
        has_prev_list.append(has_prev)

    merged_obs = np.concatenate(obs_list, axis=0)
    merged_actions = np.concatenate(act_list, axis=0)
    merged_prev_obs = np.concatenate(prev_obs_list, axis=0)
    merged_prev_actions = np.concatenate(prev_act_list, axis=0)
    merged_has_prev = np.concatenate(has_prev_list, axis=0)
    merged_weights = np.concatenate(weight_list, axis=0)

    # Stats & quality checks (informational)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {len(merged_obs)}")
    print(f"\nAction statistics:")
    print(f"Mean: {np.mean(merged_actions, axis=0)}")
    print(f"Std: {np.std(merged_actions, axis=0)}")
    print(f"Min: {np.min(merged_actions, axis=0)}")
    print(f"Max: {np.max(merged_actions, axis=0)}")
    print(f"\nWeight statistics:")
    print(f"Mean: {np.mean(merged_weights):.3f}")
    print(f"Std: {np.std(merged_weights):.3f}")
    print(f"Min: {np.min(merged_weights):.3f}")
    print(f"Max: {np.max(merged_weights):.3f}")
    print(f"Non-default weights: {np.sum(merged_weights != 1.0)} ({100*np.mean(merged_weights != 1.0):.1f}%)")
    print(f"\nData Quality Checks:")
    print(f"NaN in observations: {np.isnan(merged_obs).any()}")
    print(f"Inf in observations: {np.isinf(merged_obs).any()}")
    print(f"NaN in actions: {np.isnan(merged_actions).any()}")
    print(f"Inf in actions: {np.isinf(merged_actions).any()}")

    # Train/val split
    indices = np.random.permutation(len(merged_obs))
    split_idx = int(0.9 * len(merged_obs))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    def pack(idx):
        return {
            "obs": merged_obs[idx],
            "act": merged_actions[idx],
            "prev_obs": merged_prev_obs[idx],
            "prev_act": merged_prev_actions[idx],
            "has_prev": merged_has_prev[idx],   # shape [B, 1]
            "weights": merged_weights[idx],     # shape [B,]
        }

    train_data, val_data = pack(train_idx), pack(val_idx)
    print(f"\nSplit sizes:\nTraining samples: {len(train_idx)}\nValidation samples: {len(val_idx)}")
    return train_data, val_data


# ----------------------------
# Network & Policy
# ----------------------------
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
    """Behavioral cloning policy with temporal-aware training losses (inference shape preserved)."""
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        # ---- Spaces / dims ----
        self.obs_dim = int(observation_space.shape[0])
        self.action_dim = int(action_space.shape[0])

        # ---- Config with safe defaults ----
        hidden_dim        = int(config.get("hidden_dim", 256))
        num_hidden_layers = int(config.get("num_hidden_layers", 2))
        dropout_rate      = float(config.get("dropout_rate", 0.1))
        learning_rate     = float(config.get("learning_rate", 1e-3))
        weight_decay      = float(config.get("weight_decay", 1e-4))
        self.batch_size   = int(config.get("train_batch_size", 256))

        # loss knobs
        self.w_base   = float(config.get("w_base", 1.0))
        self.w_delta  = float(config.get("w_delta", 1.0))
        self.w_jump   = float(config.get("w_jump", 1.0))
        self.w_sat    = float(config.get("w_sat", 0.1))
        self.w_jac    = float(config.get("w_jac", 0.0))
        self.delta_max  = float(config.get("delta_max", 0.15))
        self.sat_margin = float(config.get("sat_margin", 0.9))

        # ---- Network & optimizer ----
        self.network = ImprovedBCNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_layers=num_hidden_layers,
            dropout_rate=dropout_rate
        )
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # ---- Data (optional at init) ----
        self.train = None
        self.val = None
        data_source = config.get("data_source", config.get("data_dir", None))

        if data_source:
            (self.train, self.val) = load_demonstration_data(data_source)

            # normalization statistics from train split
            self.obs_mean = np.mean(self.train["obs"], axis=0)
            self.obs_std  = np.std(self.train["obs"], axis=0) + 1e-8

            # normalize obs and prev_obs in both splits
            for split in (self.train, self.val):
                split["obs"] = (split["obs"] - self.obs_mean) / self.obs_std
                split["prev_obs"] = (split["prev_obs"] - self.obs_mean) / self.obs_std
        else:
            # Inference-safe defaults if no data loaded yet
            self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
            self.obs_std  = np.ones(self.obs_dim,  dtype=np.float32)

        # ---- Batching / scheduler ----
        self.idx = 0
        base_lr = learning_rate
        floor = 1e-6
        self._lr_floor_ratio = floor / max(base_lr, 1e-12)
        lambda_fn = lambda step: max(self._lr_floor_ratio, 1.0 - (step / 10000.0))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_fn)

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions for the current policy (unchanged inference)."""
        # Ensure numpy float32 and correct shape
        obs_batch = np.asarray(obs_batch, dtype=np.float32)
        obs_batch = (obs_batch - self.obs_mean) / self.obs_std
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_batch)
            actions = self.network(obs_tensor).cpu().numpy().astype(np.float32)
        # actions shape: (batch, action_dim); values in [-1, 1] due to tanh
        return actions, [], {}

    def _sample_train_slice(self):
        """Draw a minibatch slice; reshuffle epoch-wise for all tensors."""
        n = len(self.train["obs"])
        if self.idx + self.batch_size > n:
            perm = np.random.permutation(n)
            for k in self.train.keys():
                self.train[k] = self.train[k][perm]
            self.idx = 0
        sl = slice(self.idx, self.idx + self.batch_size)
        self.idx += self.batch_size
        return sl

    def learn_on_batch(self, samples=None):
        """Train on a batch with temporal-aware losses and sample weighting."""
        if self.train is None or self.val is None:
            raise RuntimeError("Demonstration data not loaded; provide data_source or data_dir in the config.")
        sl = self._sample_train_slice()

        obs_batch      = torch.from_numpy(self.train["obs"][sl]).float()
        act_batch      = torch.from_numpy(self.train["act"][sl]).float()
        prev_obs_batch = torch.from_numpy(self.train["prev_obs"][sl]).float()
        prev_act_batch = torch.from_numpy(self.train["prev_act"][sl]).float()
        has_prev       = torch.from_numpy(self.train["has_prev"][sl]).float()  # [B,1]
        sample_weights = torch.from_numpy(self.train["weights"][sl]).float()   # [B,]

        # Normalize weights to have mean=1 within batch (preserves relative importance)
        sample_weights = sample_weights / (sample_weights.abs().mean() + 1e-8)
        sample_weights = sample_weights.unsqueeze(1)  # [B, 1] for broadcasting

        if self.w_jac > 0.0:
            obs_batch.requires_grad_(True)

        self.network.train()
        self.optimizer.zero_grad()

        # Predictions
        pred_t = self.network(obs_batch)                 # â_t
        with torch.no_grad():                            # do not backprop through â_{t-1}
            pred_t_minus_1 = self.network(prev_obs_batch)

        # Base target loss (Huber) - weighted per sample
        per_sample_loss = torch.nn.SmoothL1Loss(reduction='none')(pred_t, act_batch)
        base_loss = torch.mean(sample_weights * per_sample_loss)

        # Delta-matching loss (weighted)
        delta_pred = pred_t - pred_t_minus_1
        delta_true = act_batch - prev_act_batch
        mask = has_prev
        if mask.ndim == 2 and pred_t.ndim == 2:
            mask = mask.expand_as(pred_t)
        delta_match_loss = torch.mean(sample_weights * mask * (delta_pred - delta_true).pow(2))

        # Jump-hinge loss (weighted)
        delta_norm = torch.linalg.norm(delta_pred, dim=1, keepdim=True)  # [B,1]
        jump_excess = torch.relu(delta_norm - self.delta_max)
        jump_hinge_loss = torch.mean(sample_weights * has_prev * (jump_excess.pow(2)))

        # Saturation margin loss (weighted)
        sat_excess = torch.relu(pred_t.abs() - self.sat_margin)
        sat_loss = torch.mean(sample_weights * sat_excess.pow(2))

        # Optional Jacobian penalty
        if self.w_jac > 0.0:
            jacobian = 0.0
            for d in range(pred_t.shape[1]):
                grads = torch.autograd.grad(
                    pred_t[:, d].sum(), obs_batch, retain_graph=True, create_graph=True
                )[0]
                jacobian = jacobian + (grads.pow(2).sum(dim=1, keepdim=True))
            jacobian_loss = jacobian.mean()
        else:
            jacobian_loss = torch.tensor(0.0, device=pred_t.device)

        loss = (
            self.w_base * base_loss +
            self.w_delta * delta_match_loss +
            self.w_jump * jump_hinge_loss +
            self.w_sat  * sat_loss +
            self.w_jac  * jacobian_loss
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- Validation metrics (no grad) ----
        self.network.eval()
        with torch.no_grad():
            v_obs      = torch.from_numpy(self.val["obs"]).float()
            v_act      = torch.from_numpy(self.val["act"]).float()
            v_prev_obs = torch.from_numpy(self.val["prev_obs"]).float()
            v_prev_act = torch.from_numpy(self.val["prev_act"]).float()
            v_has_prev = torch.from_numpy(self.val["has_prev"]).float()

            v_pred_t = self.network(v_obs)
            v_pred_tm1 = self.network(v_prev_obs)

            v_base = torch.nn.SmoothL1Loss()(v_pred_t, v_act)

            v_delta_pred = v_pred_t - v_pred_tm1
            v_delta_true = v_act - v_prev_act
            v_mask = v_has_prev.expand_as(v_pred_t)
            v_delta = torch.mean(v_mask * (v_delta_pred - v_delta_true).pow(2))

            v_delta_norm = torch.linalg.norm(v_delta_pred, dim=1, keepdim=True)
            v_jump = torch.mean(v_has_prev * torch.relu(v_delta_norm - self.delta_max).pow(2))

            v_sat = torch.mean(torch.relu(v_pred_t.abs() - self.sat_margin).pow(2))

            v_total = (
                self.w_base * v_base +
                self.w_delta * v_delta +
                self.w_jump * v_jump +
                self.w_sat  * v_sat
            )

        self.scheduler.step()

        return {
            "train_total_loss": float(loss.item()),
            "train_base": float(base_loss.item()),
            "train_delta": float(delta_match_loss.item()),
            "train_jump": float(jump_hinge_loss.item()),
            "train_sat": float(sat_loss.item()),
            "train_jac": float(jacobian_loss.item()),
            "val_total": float(v_total.item()),
            "val_base": float(v_base.item()),
            "val_delta": float(v_delta.item()),
            "val_jump": float(v_jump.item()),
            "val_sat": float(v_sat.item()),
            "cur_lr": self.optimizer.param_groups[0]["lr"],
        }


# ----------------------------
# Trainer
# ----------------------------
class ImprovedBCTrainer(Trainer):
    """Improved behavioral cloning trainer with temporal-aware losses."""
    _policy_class = ImprovedBCPolicy

    _default_config = {
        # === Required Parameters ===
        "env": None,
        "env_config": {},

        # === BC-Specific Parameters ===
        "framework": "torch",
        "train_batch_size": 256,
        "learning_rate": 1e-2,
        "hidden_dim": 256,
        "num_hidden_layers": 2,
        "data_dir": None,
        "data_source": None,
        # === New loss weights / knobs ===
        "w_base": 1.0,
        "w_delta": 1.0,
        "w_jump": 1.0,
        "w_sat": 0.1,
        "w_jac": 0.0,
        "delta_max": 0.15,
        "sat_margin": 0.9,

        # === RLlib Required Parameters ===
        "num_workers": 0,
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_gpus_per_worker": 0,
        "create_env_on_driver": True,

        # === Evaluation Parameters ===
        "evaluation_interval": None,
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": 0,
        "evaluation_config": {},
        "evaluation_parallel_to_training": False,
        "input_evaluation": [],

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
        """Initialize the trainer with a dummy env for spaces."""
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

    # --- Save/Load logic: compatible with your original pattern ---
    def save_checkpoint(self, checkpoint_dir):
        """Save a single policy.pt file in checkpoint_dir and return its full path."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "policy.pt")
        state_dict = {
            'network_state': self.policy.network.state_dict(),
            'optimizer_state': self.policy.optimizer.state_dict(),
            'scheduler_state': getattr(self.policy.scheduler, 'state_dict', lambda: {})(),
            'obs_mean': self.policy.obs_mean,
            'obs_std': self.policy.obs_std,
            'meta': {
                'obs_dim': int(self.policy.obs_dim),
                'action_dim': int(self.policy.action_dim),
                'hidden_dim': int(self.config["hidden_dim"]),
                'num_hidden_layers': int(self.config["num_hidden_layers"]),
            }
        }
        torch.save(state_dict, path)
        return path

    def load_checkpoint(self, checkpoint_path):
        """Load from either a file path or a directory containing policy.pt."""
        # If a directory is passed, append policy.pt
        if os.path.isdir(checkpoint_path):
            candidate = os.path.join(checkpoint_path, "policy.pt")
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"No policy.pt in directory: {checkpoint_path}")
            checkpoint_path = candidate

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Map to CPU by default; user can move model after restore
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Load network/opt/scheduler states when present
        self.policy.network.load_state_dict(state_dict['network_state'])
        try:
            self.policy.optimizer.load_state_dict(state_dict['optimizer_state'])
        except Exception as e:
            logger.warning(f"Optimizer state load failed (continuing): {e}")

        sched_state = state_dict.get('scheduler_state', None)
        if sched_state:
            try:
                self.policy.scheduler.load_state_dict(sched_state)
            except Exception as e:
                logger.warning(f"Scheduler state load failed (continuing): {e}")

        # Restore normalization (shape-safe)
        self.policy.obs_mean = np.asarray(state_dict.get('obs_mean', np.zeros(self.policy.obs_dim, dtype=np.float32)), dtype=np.float32)
        self.policy.obs_std  = np.asarray(state_dict.get('obs_std',  np.ones(self.policy.obs_dim,  dtype=np.float32)), dtype=np.float32)

        # Optional meta sanity check
        meta = state_dict.get('meta', {})
        if meta:
            od = int(meta.get('obs_dim', self.policy.obs_dim))
            ad = int(meta.get('action_dim', self.policy.action_dim))
            if od != self.policy.obs_dim or ad != self.policy.action_dim:
                logger.warning(f"Checkpoint dims (obs={od}, act={ad}) differ from env dims "
                               f"(obs={self.policy.obs_dim}, act={self.policy.action_dim}). Proceeding anyway.")


# ----------------------------
# CLI script
# ----------------------------
def run_bc_training(
    data_source: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    num_iters: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    hidden_dim: int = 256,
    num_layers: int = 2,
    w_delta: float = 0.0,
    w_jump: float = 0.0,
    w_sat: float = 0.0,
    delta_max: float = 0.15,
    sat_margin: float = 0.9,
    w_jac: float = 0.0,
    checkpoint_freq: int = 100,
    checkpoint_dir: str = "./checkpoints",
):
    if not ray.is_initialized():
        ray.init()
    config = {
        "env": DummyEnv,
        "env_config": {},
        "framework": "torch",
        "train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "num_hidden_layers": num_layers,
        "data_dir": None,
        "data_source": data_source,
        "w_delta": w_delta,
        "w_jump": w_jump,
        "w_sat": w_sat,
        "delta_max": delta_max,
        "sat_margin": sat_margin,
        "w_jac": w_jac,
    }
    trainer = ImprovedBCTrainer(config=config)
    best_val = float("inf")
    metrics = [
        "train_total_loss",
        "train_base",
        "train_delta",
        "train_jump",
        "train_sat",
        "val_total",
        "val_base",
        "val_delta",
        "val_jump",
        "val_sat",
        "cur_lr",
    ]
    for i in range(num_iters):
        result = trainer.train()
        def fmt(k):
            v = result.get(k, float("nan"))
            try:
                return f"{k}={v:.4f}"
            except Exception:
                return f"{k}={v}"
        if i % 50 == 0:
            print(f"Iteration {i}: " + ", ".join(fmt(k) for k in metrics))
        if result.get("val_total", float("inf")) < best_val:
            best_val = result["val_total"]
            best_dir = os.path.join(checkpoint_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            checkpoint_path = trainer.save_checkpoint(best_dir)
            print(f"New best model saved to {checkpoint_path}")
        if checkpoint_freq and i % checkpoint_freq == 0:
            iter_dir = os.path.join(checkpoint_dir, f"iter_{i}")
            os.makedirs(iter_dir, exist_ok=True)
            checkpoint_path = trainer.save_checkpoint(iter_dir)
            print(f"Checkpoint saved to {checkpoint_path}")
    return trainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing trajectory data")
    parser.add_argument("--num-iters", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--checkpoint-freq", type=int, default=100, help="Save checkpoint every N iterations")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")

    # Optional knobs for the new losses
    parser.add_argument("--w-delta", type=float, default=1.0, help="Weight for delta-matching loss")
    parser.add_argument("--w-jump", type=float, default=1.0, help="Weight for jump-hinge loss")
    parser.add_argument("--w-sat", type=float, default=0.1, help="Weight for saturation-margin loss")
    parser.add_argument("--delta-max", type=float, default=0.15, help="Per-step action change tolerance")
    parser.add_argument("--sat-margin", type=float, default=0.9, help="Saturation margin inside [-1,1]")
    parser.add_argument("--w-jac", type=float, default=0.0, help="Weight for Jacobian penalty (0 to disable)")

    args = parser.parse_args()
    run_bc_training(
        data_source=args.data_dir,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        w_delta=args.w_delta,
        w_jump=args.w_jump,
        w_sat=args.w_sat,
        delta_max=args.delta_max,
        sat_margin=args.sat_margin,
        w_jac=args.w_jac,
        checkpoint_freq=args.checkpoint_freq,
    )
