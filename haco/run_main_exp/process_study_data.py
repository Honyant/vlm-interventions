import json
with open('haco/run_main_exp/study_data/traj_metadata.json', 'r') as f:
    traj_metadata = json.load(f)

for data in traj_metadata.values():
    if data.get
        data["flawed"] = True