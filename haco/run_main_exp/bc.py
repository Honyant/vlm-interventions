import json
import numpy as np
import os
def read_traj(path):
    with open(path, 'r') as f:
        path_data = json.load(f)
    traj = path_data["trajectory"]
    obs = [t[0] for t in traj]
    actions = [t[1] for t in traj]
    # weights = [t[2] for t in traj]
    # print(np.array(obs).shape)
    # print(np.array(actions).shape)
    return (np.array(obs), np.array(actions))

dir = "/home/anthony/HACO/haco/run_main_exp/trajectory_data/"
file_list = os.listdir(dir)
traj_list = []
for file in file_list:
    traj_list.append(read_traj(dir+file))
merged_obs = np.concatenate([traj[0] for traj in traj_list], axis=0)
merged_actions = np.concatenate([traj[1] for traj in traj_list], axis=0)
print(merged_obs.shape)
print(merged_actions.shape)