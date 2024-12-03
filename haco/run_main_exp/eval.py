from labeler import TrajectoryEvaluator
if __name__ == "__main__":
    trajectory_file = "/home/anthony/HACO/haco/run_main_exp/trajectory_modified.json"
    video_dir = "/home/anthony/HACO/haco/run_main_exp/videos"
    
    # For evaluation:
    evaluator = TrajectoryEvaluator(trajectory_file, "ground_truth_labels.json")
    results = evaluator.evaluate()
    print("\nEvaluation Results:")
    print(f"Total labeled frames: {results['total_labeled_frames']}")
    print(f"Steering direction accuracy: {results['steering_accuracy']*100:.1f}%")
    print(f"Acceleration direction accuracy: {results['acceleration_accuracy']*100:.1f}%")