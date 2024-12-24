import os
import base64
import requests
from PIL import Image
import re
import json
from dotenv import load_dotenv
import copy
from pathlib import Path

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

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

def find_intervention_segments(trajectory_data, context_window=4):
    segments = []
    for i in range(len(trajectory_data)):
        if trajectory_data[i][2]:
            start_idx = max(0, i - context_window)
            end_idx = min(len(trajectory_data), i + 1)
            segments.append((start_idx, end_idx))
    
    if segments:
        merged = []
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged
    return []

def format_trajectory_data(trajectory_data, selected_indices):
    formatted_text = ""
    for i in range(0, len(selected_indices), 3):
        group = selected_indices[i:i+3]
        timestep_data = []
        for idx in group:
            if idx < len(trajectory_data):
                action = trajectory_data[idx][1]
                intervene = trajectory_data[idx][2]
                action_truncated = [round(float(x), 2) for x in action]
                timestep_data.append((f"action {idx}", action_truncated, intervene))
        
        if timestep_data:
            image_path = trajectory_data[group[0]][3]
            formatted_text += f"{timestep_data}, {image_path}\n"
    return formatted_text

def process_trajectory_file(trajectory_file, image_dir):
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
        trajectory_data = data['trajectory']
    
    segments = find_intervention_segments(trajectory_data)
    if not segments:
        print(f"No interventions found in {trajectory_file}")
        return
    
    output_dir = os.path.dirname(trajectory_file)
    output_filename = f"vlmcorrected_{os.path.basename(trajectory_file)}"
    output_path = os.path.join(output_dir, output_filename)
    
    corrected_data = copy.deepcopy(data)
    
    for start_idx, end_idx in segments:
        file_stem = Path(trajectory_file).stem
        match = re.search(r'trajectory_ckpt_(\d+)_episode_(\d+)', file_stem)
        if match:
            ckpt_num, episode_num = match.groups()
            prefix = f"frame_{ckpt_num}_{episode_num}"
        else:
            print(f"Could not parse filename format for {trajectory_file}")
            continue
        
        images = [img for img in os.listdir(image_dir) if img.startswith(prefix) and img.endswith(".jpg")]
        images.sort(key=natural_sort_key)
        
        selected_images = images[start_idx:end_idx:3]
        image_paths = [os.path.join(image_dir, img) for img in selected_images]
        grid_path = os.path.join(image_dir, "grid_image.jpg")
        create_image_grid(image_paths, grid_path)
        base64_image = encode_image(grid_path)
        
        selected_indices = range(start_idx, end_idx)
        trajectory_text = format_trajectory_data(trajectory_data, selected_indices)
        
        input_text = f"""Trajectory data:\n{trajectory_text}\n\n
Analyze the sequence of driving actions before an intervention to avoid a truck collision. For each action frame:

1. Provide a brief text description suggesting better pre-intervention actions
2. Output a JSON with index and (before, after) pairs for steering [-1,1] and acceleration [-1,1] values
- Use negative acceleration values for deceleration
- Include 3 significant figures
- Start from index {start_idx}
- Include all consecutive actions until intervention

Example format:
```json
[
  {{
    "index": 192,
    "steering": [0.071, 0.150],
    "acceleration": [0.999, 0.500]
  }},
  ...
]
```
"""
        
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
        response_content = response.json()['choices'][0]['message']['content']
        
        json_str = response_content[response_content.find('['):response_content.rfind(']')+1]
        corrections = json.loads(json_str)
        
        for correction in corrections:
            idx = correction['index']
            corrected_data['trajectory'][idx][1] = correction['steering'] + correction['acceleration']
    
    with open(output_path, 'w') as f:
        json.dump(corrected_data, f, indent=2)
    
    print(f"Processed {len(segments)} intervention segments. Saved to {output_path}")

def process_directory(trajectory_dir, image_dir):
    for filename in os.listdir(trajectory_dir):
        if filename.startswith("trajectory_") and filename.endswith(".json"):
            filepath = os.path.join(trajectory_dir, filename)
            print(f"Processing {filepath}...")
            process_trajectory_file(filepath, image_dir)

trajectory_dir = "/home/anthony/HACO/haco/run_main_exp/trajectory_data"
image_dir = "/home/anthony/HACO/haco/run_main_exp/videos"
process_directory(trajectory_dir, image_dir)