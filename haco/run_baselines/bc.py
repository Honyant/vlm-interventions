from __future__ import print_function

import os
import time
import torch
import numpy as np
from haco.algo.HG_Dagger.exp_saver import Experiment
from haco.algo.HG_Dagger.model import Ensemble
from haco.algo.HG_Dagger.utils import load_human_data, train_model, evaluation
from haco.utils.config import baseline_eval_config, baseline_train_config
from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
# hyperparameters
NUM_ITS = 5
STEP_PER_ITER = 5000
learning_rate = 5e-4
batch_size = 256

evaluation_episode_num = 30
num_sgd_epoch = 10000  # sgd epoch on data set
device = "cpu"

# training env_config/test env config
training_config = baseline_train_config
training_config["use_render"] = True
training_config["manual_control"] = True
eval_config = baseline_eval_config


if __name__ == "__main__":
    tm = time.localtime(time.time())
    tm_stamp = "%s-%s-%s-%s-%s-%s" % (tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
    log_dir = os.path.join(
        "bc_lr_{}_bs_{}_sgd_iter_{}_iter_batch_{}".format(learning_rate, batch_size, num_sgd_epoch, STEP_PER_ITER), tm_stamp)
    exp_log = Experiment()
    exp_log.init(log_dir=log_dir)
    model_save_path = os.path.join(log_dir, "bc_models")
    os.mkdir(model_save_path)
    eval_env = HumanInTheLoopEnv(eval_config)

    # load and merge expert data from first and second type interventions
    samples1 = load_human_data("/home/anthony/HACO/haco/utils/human_traj_file1_4.json", data_usage=30)
    samples2 = load_human_data("/home/anthony/HACO/haco/utils/human_traj_file2_4.json", data_usage=30)
    
    # Merge samples
    X_train = np.concatenate([samples1["state"], samples2["state"]], axis=0)
    y_train = np.concatenate([samples1["action"], samples2["action"]], axis=0)
    
    # Create target indices
    target_indices = [0] * len(samples1["state"]) + [1] * len(samples2["state"])

    # obs shape is 259, action is 2
    obs_shape = [259]
    action_shape = [1,2]

    # train model
    agent = Ensemble(obs_shape, action_shape, device=device, hidden_sizes=(259, 256)).to(device).float()
    
    train_model(agent, X_train, y_train,
                os.path.join(model_save_path, "model_{}.pth".format(0)),
                target_indices=target_indices,
                num_epochs=num_sgd_epoch,
                batch_size=batch_size,
                learning_rate=learning_rate,
                exp_log=exp_log)
    evaluation(eval_env, agent, evaluation_episode_num=evaluation_episode_num, exp_log=exp_log)
    
    exp_log.end_iteration(0)