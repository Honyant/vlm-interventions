

import os
import base64
import requests
from PIL import Image
import re
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

image_dir = "/home/anthony/HACO/haco/run_main_exp/videos"

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_grid(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    max_height = max(heights)
    grid = Image.new('RGB', (max_width * 3, max_height * 3))
    for index, image in enumerate(images):
        row = index // 3
        col = index % 3
        grid.paste(image, (col * max_width, row * max_height))
    grid.save(output_path)

def format_trajectory_data(trajectory_data, selected_indices):
    formatted_text = ""
    for i in range(0, len(selected_indices), 3):
        group = selected_indices[i:i+3]
        timestep_data = []
        for idx in group:
            if idx < len(trajectory_data):
                obs = trajectory_data[idx][0]
                action = trajectory_data[idx][1]
                intervene = trajectory_data[idx][2]
                # Truncate floats to 2 decimal places
                obs_truncated = [round(float(x), 2) for x in obs]
                action_truncated = [round(float(x), 2) for x in action]
                timestep_data.append((f"action {idx}", action_truncated, intervene))
                # timestep_data.append((f"action {idx}", obs_truncated, action_truncated, intervene))

        if timestep_data:
            image_path = trajectory_data[group[0]][3]  # Use the image path from the first entry in group
            formatted_text += f"{timestep_data}, {image_path}\n"
    return formatted_text

def send_images(prefix, base_index, interval, num_images, trajectory_file):
    # Load trajectory data
    with open(trajectory_file, 'r') as f:
        trajectory_data = json.load(f)['trajectory']

    images = [img for img in os.listdir(image_dir) if img.startswith(prefix) and img.endswith(".jpg")]
    images.sort(key=natural_sort_key)
    selected_images = images[base_index:base_index + interval * num_images:interval]
    image_paths = [os.path.join(image_dir, img) for img in selected_images]
    
    # Get corresponding indices from trajectory data
    selected_indices = range(base_index, base_index + interval * num_images, 1)
    trajectory_text = format_trajectory_data(trajectory_data, selected_indices)
    
    grid_path = os.path.join(image_dir, "grid_image.jpg")
    create_image_grid(image_paths, grid_path)
    base64_image = encode_image(grid_path)
    

    input_text = f"""Trajectory data:\n{trajectory_text}\n\n
Analyze the sequence of driving actions before an intervention to avoid a truck collision. For each action frame (every 3rd frame):

1. Provide a brief text description suggesting better pre-intervention actions
2. Output a JSON with (before, after) pairs for steering [-1,1] and acceleration [-1,1] values
- Use negative acceleration values for deceleration
- Include 3 significant figures
- Start from {base_index}
- Include all consecutive actions until intervention

Example format requested, (all in a single json list):
```json
{{
"steering": [before, after],
"acceleration": [before, after]
}}
```
"""
    print(input_text)
    # exit()
    

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": input_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 700
    }

    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json()['choices'][0]['message']['content'])

trajectory_file = "/home/anthony/HACO/haco/run_main_exp/trajectory_data/trajectory_ckpt_53_episode_0.json"
send_images(prefix="frame_53_0", base_index=192, interval=3, num_images=9, trajectory_file=trajectory_file)