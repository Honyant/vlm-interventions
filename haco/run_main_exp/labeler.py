import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import json
import numpy as np
import os

class TrajectoryLabeler:
    def __init__(self, trajectory_file, video_dir):
        self.root = tk.Tk()
        self.root.title("Trajectory Labeler")
        self.root.geometry("3200x2400")  # 2x larger window

        # Apply 2x scaling for fonts and paddings
        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 24))  # Larger labels
        style.configure('TButton', font=('Helvetica', 24), padding=20)
        style.configure('TEntry', font=('Helvetica', 24))  # Larger entries
        style.configure('TScale', font=('Helvetica', 24))  # Larger scale
        
        
        # Load trajectory data
        with open(trajectory_file, 'r') as f:
            self.data = json.load(f)
        self.trajectory = self.data['trajectory']
        
        # Filter for takeover frames
        self.takeover_indices = [
            i for i, (_, _, takeover, _) in enumerate(self.trajectory) 
            if True # takeover
        ]
        self.current_idx = 0  # Index in takeover_indices
        self.video_dir = video_dir
        
        # Store labels
        self.labels = {}  # {idx: {'gt_steering': float, 'gt_acceleration': float}}
        
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top info panel
        info_panel = ttk.Frame(main_frame)
        info_panel.pack(fill=tk.X, pady=10)
        
        ttk.Label(info_panel, text="Frame:").pack(side=tk.LEFT)
        
        # Timeline slider
        self.timeline = ttk.Scale(
            info_panel, 
            from_=0, 
            to=len(self.takeover_indices)-1,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Frame counter
        self.frame_counter = ttk.Label(info_panel, text="0/0")
        self.frame_counter.pack(side=tk.LEFT)
        
        # Image display (larger)
        self.img_label = ttk.Label(main_frame)
        self.img_label.pack(pady=10)
        
        # Action info panel
        info_frame = ttk.LabelFrame(main_frame, text="Action Info", padding="10")
        info_frame.pack(fill=tk.X, pady=10)
        
        # Original actions
        action_frame = ttk.Frame(info_frame)
        action_frame.pack(fill=tk.X)
        
        ttk.Label(action_frame, text="Original Actions:").pack(side=tk.LEFT)
        self.action_var = tk.StringVar()
        ttk.Label(action_frame, textvariable=self.action_var).pack(side=tk.LEFT, padx=10)
        
        # Ground truth inputs
        gt_frame = ttk.Frame(info_frame)
        gt_frame.pack(fill=tk.X, pady=10)
        
        # Steering input
        ttk.Label(gt_frame, text="Ground Truth Steering:").pack(side=tk.LEFT)
        self.gt_steering = ttk.Entry(gt_frame, width=10)
        self.gt_steering.pack(side=tk.LEFT, padx=10)
        
        # Acceleration input
        ttk.Label(gt_frame, text="Ground Truth Acceleration:").pack(side=tk.LEFT)
        self.gt_accel = ttk.Entry(gt_frame, width=10)
        self.gt_accel.pack(side=tk.LEFT, padx=10)
        
        # Navigation and save
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(pady=10)
        
        ttk.Button(nav_frame, text="Previous", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Save", command=self.save_labels).pack(side=tk.LEFT, padx=5)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Return>', lambda e: self.save_current_labels())
        
        hotkey_frame = ttk.LabelFrame(main_frame, text="Hotkeys", padding="10")
        hotkey_frame.pack(fill=tk.X, pady=10)
        
        instructions = """
        Steering Hotkeys:    'q': -1    'w': 0    'e': 1
        Acceleration Hotkeys: 'a': -1    's': 0    'd': 1
        Navigation:          '←': Previous    '→': Next
        """
        ttk.Label(hotkey_frame, text=instructions).pack()

        # Add quit button
        quit_frame = ttk.Frame(main_frame)
        quit_frame.pack(pady=20)
        
        quit_button = ttk.Button(
            quit_frame, 
            text="QUIT AND SAVE", 
            command=self.quit_labeling,
            style='Large.TButton'
        )
        quit_button.pack(pady=10)

        # Create custom style for larger button
        style = ttk.Style()
        style.configure('Large.TButton', font=('Helvetica', 14, 'bold'), padding=10)

        # Keyboard bindings
        self.root.bind('q', lambda e: self.set_label('steering', -1))
        self.root.bind('w', lambda e: self.set_label('steering', 0))
        self.root.bind('e', lambda e: self.set_label('steering', 1))
        self.root.bind('a', lambda e: self.set_label('acceleration', -1))
        self.root.bind('s', lambda e: self.set_label('acceleration', 0))
        self.root.bind('d', lambda e: self.set_label('acceleration', 1))
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        
        self.update_display()
        
    def on_slider_change(self, value):
        self.current_idx = int(float(value))
        self.update_display()
        
    def update_display(self):
        # Get actual trajectory index
        traj_idx = self.takeover_indices[self.current_idx]
        
        # Update frame counter
        self.frame_counter.config(
            text=f"{self.current_idx + 1}/{len(self.takeover_indices)} (Trajectory idx: {traj_idx})"
        )
        
        # Update image
        frame_path = self.trajectory[traj_idx][3]
        if frame_path and os.path.exists(frame_path):
            img = Image.open(frame_path)
            # Resize while maintaining aspect ratio
            display_size = (1200, 800)
            img.thumbnail(display_size, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.img_label.configure(image=photo)
            self.img_label.image = photo
            
        # Update action info
        _, action, _, _ = self.trajectory[traj_idx]
        self.action_var.set(f"Steering: {action[0]:.4f}, Acceleration: {action[1]:.4f}")
        
        # Update ground truth inputs if already labeled
        if traj_idx in self.labels:
            self.gt_steering.delete(0, tk.END)
            self.gt_steering.insert(0, str(self.labels[traj_idx]['gt_steering']))
            self.gt_accel.delete(0, tk.END)
            self.gt_accel.insert(0, str(self.labels[traj_idx]['gt_acceleration']))
        else:
            self.gt_steering.delete(0, tk.END)
            self.gt_accel.delete(0, tk.END)
    
    def set_label(self, control_type, value):
        """Set label value without automatic frame advancement"""
        if control_type == 'steering':
            self.gt_steering.delete(0, tk.END)
            self.gt_steering.insert(0, str(float(value)))
        else:  # acceleration
            self.gt_accel.delete(0, tk.END)
            self.gt_accel.insert(0, str(float(value)))

    def next_frame(self):
        # """Only advance if both values are set"""
        # if not self.gt_steering.get() or not self.gt_accel.get():
        #     print("Please set both steering and acceleration values before advancing")
        #     return
            
        self.save_current_labels()
        if self.current_idx < len(self.takeover_indices) - 1:
            self.current_idx += 1
            self.timeline.set(self.current_idx)
            self.update_display()

    def prev_frame(self):
        # """Only go back if both values are set for current frame"""
        # if not self.gt_steering.get() or not self.gt_accel.get():
        #     print("Please set both steering and acceleration values before moving to previous frame")
        #     return
            
        self.save_current_labels()
        if self.current_idx > 0:
            self.current_idx -= 1
            self.timeline.set(self.current_idx)
            self.update_display()
    
    def save_current_labels(self):
        try:
            gt_steer = float(self.gt_steering.get()) if self.gt_steering.get() else None
            gt_accel = float(self.gt_accel.get()) if self.gt_accel.get() else None
            
            if gt_steer is not None or gt_accel is not None:
                traj_idx = self.takeover_indices[self.current_idx]
                self.labels[traj_idx] = {
                    'gt_steering': gt_steer,
                    'gt_acceleration': gt_accel
                }
        except ValueError:
            print("Invalid input - must be numbers")
            
    def quit_labeling(self):
        """Save current labels and quit the application"""
        self.save_labels()
        self.root.quit()
        self.root.destroy()
    
    def save_labels(self):
        self.save_current_labels()
        output = {
            'trajectory_file': os.path.basename(trajectory_file),
            'labels': self.labels
        }
        print("Saving labels to ground_truth_labels.json")
        with open('ground_truth_labels.json', 'w') as f:
            json.dump(output, f)
        print("Labels saved to ground_truth_labels.json")
    
    def run(self):
        self.root.mainloop()

class TrajectoryEvaluator:
    def __init__(self, trajectory_file, ground_truth_file):
        # Load trajectory
        with open(trajectory_file, 'r') as f:
            self.data = json.load(f)
        self.trajectory = self.data['trajectory']
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
    
    def evaluate(self):
        results = {
            'steering_correct_direction': 0,
            'acceleration_correct_direction': 0,
            'total_labeled_frames': 0,
            'steering_details': [],
            'acceleration_details': []
        }
        
        for idx, labels in self.ground_truth['labels'].items():
            idx = int(idx)
            original_action = self.trajectory[idx][1]
            
            # Check steering direction
            if labels['gt_steering'] is not None:
                results['total_labeled_frames'] += 1
                orig_steer = original_action[0]
                gt_steer = labels['gt_steering']
                
                # Check if they're on the same side of 0, or if the gt is 0, then it doesn't matter
                correct_direction = (orig_steer * gt_steer > 0) or (gt_steer == 0)
                if correct_direction:
                    results['steering_correct_direction'] += 1
                    
                results['steering_details'].append({
                    'frame': idx,
                    'original': orig_steer,
                    'ground_truth': gt_steer,
                    'correct': correct_direction
                })
            
            # Check acceleration direction
            if labels['gt_acceleration'] is not None:
                orig_accel = original_action[1]
                gt_accel = labels['gt_acceleration']
                
                correct_direction = (orig_accel * gt_accel > 0) or (orig_accel == gt_accel == 0)
                if correct_direction:
                    results['acceleration_correct_direction'] += 1
                    
                results['acceleration_details'].append({
                    'frame': idx,
                    'original': orig_accel,
                    'ground_truth': gt_accel,
                    'correct': correct_direction
                })
        
        # Calculate percentages
        if results['total_labeled_frames'] > 0:
            results['steering_accuracy'] = results['steering_correct_direction'] / results['total_labeled_frames']
            results['acceleration_accuracy'] = results['acceleration_correct_direction'] / results['total_labeled_frames']
            
        return results

# Example usage:
if __name__ == "__main__":
    trajectory_file = "/home/anthony/HACO/haco/run_main_exp/trajectory_data/trajectory_ckpt_53_episode_0.json"
    video_dir = "/home/anthony/HACO/haco/run_main_exp/videos"
    
    # For labeling:
    labeler = TrajectoryLabeler(trajectory_file, video_dir)
    labeler.run()
    
    # For evaluation:
    evaluator = TrajectoryEvaluator(trajectory_file, "ground_truth_labels.json")
    results = evaluator.evaluate()
    print("\nEvaluation Results:")
    print(f"Total labeled frames: {results['total_labeled_frames']}")
    print(f"Steering direction accuracy: {results['steering_accuracy']*100:.1f}%")
    print(f"Acceleration direction accuracy: {results['acceleration_accuracy']*100:.1f}%")