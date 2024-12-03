import json

# Original reference data with the new values
reference_data = [
    {
        "id": 192,
        "steering": [0.02, 0.02],
        "acceleration": [0.96, -0.5]
    },
    {
        "id": 193,
        "steering": [-0.03, -0.03],
        "acceleration": [0.96, -0.5]
    },
    {
        "id": 194,
        "steering": [0.12, 0.12],
        "acceleration": [0.94, -0.5]
    },
    {
        "id": 195,
        "steering": [-0.06, -0.06],
        "acceleration": [1.0, -0.5]
    },
    {
        "id": 196,
        "steering": [0.14, 0.14],
        "acceleration": [0.96, -0.5]
    },
    {
        "id": 197,
        "steering": [-0.06, -0.06],
        "acceleration": [0.99, -0.5]
    },
    {
        "id": 198,
        "steering": [0.07, 0.07],
        "acceleration": [0.99, -0.5]
    },
    {
        "id": 199,
        "steering": [-0.11, -0.11],
        "acceleration": [0.99, -0.5]
    },
    {
        "id": 200,
        "steering": [-0.11, -0.11],
        "acceleration": [0.99, -0.5]
    },
    {
        "id": 201,
        "steering": [-0.17, -0.17],
        "acceleration": [1.0, -0.5]
    },
    {
        "id": 202,
        "steering": [-0.69, -0.69],
        "acceleration": [0.98, -0.5]
    }
]

# Read the original trajectory file
with open('/home/anthony/HACO/haco/run_main_exp/trajectory_data/trajectory_ckpt_53_episode_0.json', 'r') as f:
    data = json.load(f)

# For each trajectory entry
for idx, entry in enumerate(data['trajectory']):
    ref_idx = idx - 192  # Since we start at ID 192
    if idx < 192:
        continue
    if idx <= 201:  # Make sure we don't go beyond our reference data
        # Get the "after" values from reference data
        new_steering = reference_data[ref_idx]['steering'][1]
        new_accel = reference_data[ref_idx]['acceleration'][1]
        # Replace the second subarray with new values
        entry[1] = [new_steering, new_accel]

# Write the modified data to a new file
with open('trajectory_modified.json', 'w') as f:
    json.dump(data, f)