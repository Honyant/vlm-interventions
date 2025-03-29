import os
import base64
import requests
from PIL import Image
import re
import json
import math
# from dotenv import load_dotenv
import copy
from pathlib import Path

# Load environment variables (including OPENAI_API_KEY)
# load_dotenv()

IGNORE_FAILED_TRAJECTORIES = True

# --- Helper Functions ---

def natural_sort_key(s):
    """Sort key for natural string sorting."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def encode_image(image_path):
    """Encode an image file in base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_image_grid(image_paths, output_path):
    """
    Create a grid image from a list of image paths.
    This function uses PIL to create a grid with 3 columns.
    """
    images = [Image.open(path) for path in image_paths]
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)
    num_cols = 3
    num_rows = (len(images) + num_cols - 1) // num_cols
    grid = Image.new('RGB', (max_width * num_cols, max_height * num_rows))
    for index, image in enumerate(images):
        row = index // num_cols
        col = index % num_cols
        grid.paste(image, (col * max_width, row * max_height))
    grid.save(output_path)

def find_intervention_segments(trajectory_data):
    """
    Identify contiguous segments where intervention_occuring is True.
    Each trajectory record is assumed to be a list in the format:
      [observation, action, intervention_occuring, image_file]
    Returns a list of tuples: (start_index, end_index) where end_index is exclusive.
    """
    segments = []
    i = 0
    n = len(trajectory_data)
    while i < n:
        if trajectory_data[i][2] is True:
            start = i
            while i < n and trajectory_data[i][2] is True:
                i += 1
            end = i
            if (end - start) > 2:
                segments.append((start, end))
        else:
            i += 1
    return segments

def get_transcript_for_range(subs, start_time, end_time):
    """
    Given a list of subtitle entries (from pysrt) and a start/end time in seconds,
    return a concatenated transcript string of all subtitles overlapping that range.
    """
    transcript_lines = []
    for sub in subs:
        sub_start = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
        sub_end = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
        if sub_end >= start_time and sub_start <= end_time:
            transcript_lines.append(sub.text)
    return "\n".join(transcript_lines)

# --- Main Processing Function ---

def process_trajectory_file(trajectory_file, image_dir, audio_dir, corrected_dir, timesteps_per_second=10, context_steps=10):
    """
    Process a single trajectory file:
      - Load trajectory JSON.
      - Detect intervention segments.
      - For each segment, select a context window (10 steps before + intervention period),
        extract corresponding transcript (if available) and image grid,
        query GPT-4o for corrections, and update the trajectory actions.
      - Save the corrected trajectory in the corrected_dir.
    """
    # print(f"Loading trajectory file: {trajectory_file}")
    with open(trajectory_file, 'r') as f:
        data = json.load(f)
    trajectory_data = data.get("trajectory", [])
    if not trajectory_data:
        print(f"No trajectory data found in {trajectory_file}")
        return
    if IGNORE_FAILED_TRAJECTORIES and data.get("metrics", {}).get("success_rate") == 0:
        print(f"Skipping {os.path.basename(trajectory_file)}: failed trajectory.")
        return
    return
    segments = find_intervention_segments(trajectory_data)
    if not segments:
        print(f"No interventions found in {os.path.basename(trajectory_file)}. Copying file without changes.")
        os.makedirs(corrected_dir, exist_ok=True)
        corrected_filepath = os.path.join(corrected_dir, os.path.basename(trajectory_file))
        with open(corrected_filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return

    # Attempt to load an SRT transcript file (assumed to have same basename with .srt extension)
    base_name = os.path.splitext(trajectory_file)[0]
    base_filename = os.path.basename(base_name)
    srt_file = os.path.join(audio_dir, base_filename + ".srt")
    # print(f"SRT file: {srt_file}")
    subs = None
    if os.path.exists(srt_file):
        try:
            import pysrt
            subs = pysrt.open(srt_file)
            # print(f"Loaded transcript from {srt_file}")
        except Exception as e:
            print(f"Error loading SRT file {srt_file}: {e}")
    else:
        print(f"No transcript found for {trajectory_file}. Proceeding without transcript context.")

    # Process each intervention segment.
    for (seg_start, seg_end) in segments:
        # Define the context window: 10 steps before intervention start and the entire intervention.
        context_start = max(0, seg_start - context_steps)
        context_end = seg_end  # end index is exclusive

        # Build trajectory text context.
        trajectory_text_lines = []
        for idx in range(context_start, context_end):
            action = trajectory_data[idx][1]
            trajectory_text_lines.append(f"Index {idx}: action={action}, intervention={trajectory_data[idx][2]}")
        trajectory_text = "\n".join(trajectory_text_lines)

        # Extract transcript for the corresponding time window (if available).
        if subs is not None:
            start_time = context_start / timesteps_per_second
            end_time = context_end / timesteps_per_second
            transcript_text = get_transcript_for_range(subs, start_time, end_time)
        else:
            transcript_text = "No transcript available."

        # Gather images from the context window.
        image_paths = []
        for idx in range(context_start, context_end):
            image_file = trajectory_data[idx][3]
            if os.path.exists(image_file):
                image_paths.append(image_file)
            else:
                print(f"Warning: Image file {image_file} not found for index {idx}.")
        if len(image_paths) == 0:
            print("No images found for this context; skipping image grid creation.")
            base64_image = ""
        else:
            # Select a subset (every 3rd image) for the grid.
            selected_images = image_paths[::3]
            grid_filename = f"grid_{Path(base_name).stem}_{seg_start}_{seg_end}.jpg"
            grid_path = os.path.join(image_dir, grid_filename)
            try:
                create_image_grid(selected_images, grid_path)
                base64_image = encode_image(grid_path)
            except Exception as e:
                print(f"Error creating image grid: {e}")
                base64_image = ""

        # Construct the prompt for GPT-4o.
        input_text = f"""Trajectory data for context window (indices {context_start} to {context_end}):
{trajectory_text}

Transcript (timestamps in seconds, with timesteps_per_second = {timesteps_per_second}):
{transcript_text}

Analyze the sequence of driving actions before and during the intervention. For each action frame in this context window:
1. Provide a brief text description suggesting better pre-intervention actions.
2. Output a JSON array with objects containing the index and corrected (steering, acceleration) pairs for that index.
- Use negative acceleration values for deceleration.
- Include 3 significant figures.
- Start from index {context_start}.
- Cover all time steps in this context window.
- If the intervention is not happening, just return the original action.
Example format YOU MUST FOLLOW:
```json
[
  {{
    "index": 192,
    "steering": [0.071, 0.150],
    "acceleration": [0.999, 0.500]
  }},
  ...
]
```"""
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
            "max_tokens": 10000
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        
        max_attempts = 5
        attempt = 0
        success = False
        
        while attempt < max_attempts and not success:
            try:
                basename = os.path.basename(trajectory_file)
                print(f"Attempt {attempt+1}/{max_attempts}: Querying GPT-4o for intervention segment {seg_start}-{seg_end} in {basename}...")
                print(f"Payload: {input_text}")
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                response_json = response.json()
                response_content = response_json['choices'][0]['message']['content']
                print(f"Response: {response_content}")
                # Extract JSON array from the response.
                json_start = response_content.find('[')
                json_end = response_content.rfind(']') + 1
                json_str = response_content[json_start:json_end]
                corrections = json.loads(json_str)
                # Apply the corrections to the trajectory data.
                for correction in corrections:
                    idx = correction['index']
                    new_steering = correction['steering']
                    new_acceleration = correction['acceleration']
                    print(f"New steering: {new_steering}, new acceleration: {new_acceleration}")
                    
                    if (isinstance(new_steering, list) and isinstance(new_acceleration, list) and
                        len(new_steering) == 2 and len(new_acceleration) == 2):
                        valid_values = True
                        for val in new_steering + new_acceleration:
                            if not isinstance(val, (int, float)) or not (-1 <= val <= 1) or math.isnan(val):
                                valid_values = False
                                break
                        if valid_values:
                            trajectory_data[idx][1] = new_steering + new_acceleration  # Concatenate lists
                        else:
                            raise ValueError(f"Invalid values in steering {new_steering} or acceleration {new_acceleration} for index {idx}")
                    else:
                        raise ValueError(f"Invalid steering or acceleration format for index {idx}: {new_steering} and {new_acceleration}")
                
                print(f"Applied corrections for segment {seg_start}-{seg_end} in {trajectory_file}.")
                success = True
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt}/{max_attempts} failed for segment {seg_start}-{seg_end} in {trajectory_file}: {e}")
                if attempt >= max_attempts:
                    print(f"All {max_attempts} attempts failed for segment {seg_start}-{seg_end} in {trajectory_file}. Exiting.")
                    exit()
                else:
                    print(f"Retrying... ({attempt+1}/{max_attempts})")

    # Save the corrected trajectory into the corrected directory.
    os.makedirs(corrected_dir, exist_ok=True)
    corrected_filename = os.path.basename(trajectory_file)
    corrected_filepath = os.path.join(corrected_dir, corrected_filename)
    with open(corrected_filepath, 'w') as f:
        json.dump(data, f, indent=2)
    # print(f"Saved corrected trajectory to {corrected_filepath}")
    print(f"Saved corrected trajectory to {os.path.basename(corrected_filepath)}")

def process_directory(trajectory_dir, image_dir, audio_dir, corrected_dir):
    """
    Process all trajectory files in the specified directory.
    Only files starting with 'trajectory_' and ending with '.json' will be processed.
    """
    for filename in os.listdir(trajectory_dir):
        if filename.startswith("trajectory_") and filename.endswith(".json"):
            filepath = os.path.join(trajectory_dir, filename)
            # print(f"Processing {filepath}...")
            process_trajectory_file(filepath, image_dir, audio_dir, corrected_dir)

# --- Main Execution ---

if __name__ == "__main__":
    # Define directories. Adjust these paths as needed.
    base_dir = "/home/anthony/vlm-interventions/haco/run_main_exp"
    trajectory_dir = os.path.join(base_dir, "trajectory_data")
    image_dir = os.path.join(base_dir, "videos")
    audio_dir = os.path.join(base_dir, "audio_recordings")
    corrected_dir = os.path.join(base_dir, "corrected_trajectories")
    
    # Process all trajectory files.
    process_directory(trajectory_dir, image_dir, audio_dir, corrected_dir)
