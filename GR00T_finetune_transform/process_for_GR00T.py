# -*- coding: utf-8 -*-
"""
Data Processing Module for GR00T Fine-tuning

This script provides utility functions for preprocessing datasets
before fine-tuning the GR00T model, including data validation,
normalization, and format conversion.

Author: VagRant2333
Email: helloforrest23@gmail.com
Created: August 18, 2025
"""


import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def generate_task_mapping(episodes_jsonl_path: Path, tasks_jsonl_path: Path):
    """
    Generates a tasks.jsonl file from episodes.jsonl if it doesn't exist.
    This file maps task description strings to integer indices.

    Args:
        episodes_jsonl_path (Path): Path to the episodes.jsonl file.
        tasks_jsonl_path (Path): Path where the tasks.jsonl file will be saved.
    """
    if tasks_jsonl_path.exists():
        print(f"'{tasks_jsonl_path}' already exists. Skipping generation.")
        return

    print(f"Generating '{tasks_jsonl_path}'...")
    tasks_set = set()
    with open(episodes_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                episode_data = json.loads(line)
                # Ensure 'tasks' key exists and is a list
                if "tasks" in episode_data and isinstance(episode_data["tasks"], list):
                    tasks_set.update(episode_data["tasks"])
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from a line in {episodes_jsonl_path}")

    tasks_list = sorted(list(tasks_set))  # Sort for consistent indexing

    tasks_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tasks_jsonl_path, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks_list):
            json.dump({"task_index": i, "task": task}, f)
            f.write("\n")
    print(f"Successfully generated '{tasks_jsonl_path}' with {len(tasks_list)} unique tasks.")

def load_task_mapping(tasks_jsonl_path: Path) -> dict:
    """
    Loads the task string to task index mapping from tasks.jsonl.

    Args:
        tasks_jsonl_path (Path): Path to the tasks.jsonl file.

    Returns:
        dict: A dictionary mapping task descriptions to their indices.
    """
    if not tasks_jsonl_path.exists():
        raise FileNotFoundError(
            f"Task mapping file not found at '{tasks_jsonl_path}'. "
            "Please generate it first."
        )
    
    task_to_idx = {}
    with open(tasks_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                task_data = json.loads(line)
                task_to_idx[task_data["task"]] = task_data["task_index"]
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from a line in {tasks_jsonl_path}")
    return task_to_idx

def process_dataset(data_dir: Path, episodes_jsonl_path: Path, task_to_idx: dict):
    """
    Processes all Parquet files in the dataset directory. It broadcasts the task
    description to each frame and then replaces the string with its corresponding
    integer index.

    Args:
        data_dir (Path): The directory containing the episode Parquet files.
        episodes_jsonl_path (Path): Path to the episodes.jsonl file.
        task_to_idx (dict): A dictionary mapping task descriptions to their indices.
    """
    print(f"Loading episode information from '{episodes_jsonl_path}'...")
    episode_tasks = {}
    with open(episodes_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                ep = json.loads(line)
                # Assuming one primary task per episode as per the original logic.
                # GR00T expects a single task description per frame.
                if "episode_index" in ep and ep.get("tasks"):
                    episode_tasks[ep["episode_index"]] = ep["tasks"][0]
            except (json.JSONDecodeError, IndexError):
                print(f"Warning: Skipping a malformed or task-less line in {episodes_jsonl_path}")

    parquet_files = sorted(list(data_dir.glob("episode_*.parquet")))
    if not parquet_files:
        print(f"Warning: No Parquet files found in '{data_dir}'. Please check the path.")
        return

    print(f"Processing {len(parquet_files)} Parquet files in '{data_dir}'...")
    for parquet_file in tqdm(parquet_files, desc="Updating Parquet files"):
        try:
            df = pd.read_parquet(parquet_file)

            if df.empty:
                print(f"Warning: Skipping empty file {parquet_file}")
                continue

            # --- Step 1: Broadcast task from episode metadata to each frame ---
            ep_idx = df["episode_index"].iloc[0]
            if ep_idx not in episode_tasks:
                print(f"Warning: No task found for episode_index {ep_idx} in {parquet_file}. Skipping.")
                continue
            
            task_description = episode_tasks[ep_idx]
            # This new column will hold the task description string for every frame.
            # GR00T expects this column for loading annotations.
            df["annotation.tasks"] = task_description

            # --- Step 2: Replace the task string with its integer index ---
            if task_description not in task_to_idx:
                print(f"Warning: Task '{task_description}' not in task mapping for {parquet_file}. Skipping conversion.")
                continue
            
            task_index = task_to_idx[task_description]
            df["annotation.tasks"] = task_index
            
            # Ensure the data type is suitable for GR00T (integer)
            df["annotation.tasks"] = df["annotation.tasks"].astype(int)

            # Save the modified DataFrame, overwriting the original file.
            df.to_parquet(parquet_file, index=False)

        except Exception as e:
            print(f"Error processing {parquet_file}: {e}")

    print("Dataset processing complete.")


def main():
    """Main function to parse arguments and run the preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess a fourier-lerobot dataset for GR00T fine-tuning. "
                    "This script adds a task annotation column to each frame in the "
                    "Parquet files and converts task descriptions to integer indices."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the dataset chunks (e.g., 'output_dataset/data/'). "
             "The script will search for 'chunk-*' subdirectories here."
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        required=True,
        help="Path to the directory containing the 'episodes.jsonl' metadata file (e.g., 'output_dataset/meta/')."
    )
    args = parser.parse_args()

    root_data_dir = Path(args.data_dir)
    meta_dir = Path(args.meta_dir)

    episodes_jsonl = meta_dir / "episodes.jsonl"
    tasks_jsonl = meta_dir / "tasks.jsonl"

    if not episodes_jsonl.is_file():
        print(f"Error: 'episodes.jsonl' not found in '{meta_dir}'. Please check the path.")
        return

    # Step 1: Generate tasks.jsonl if it doesn't exist
    generate_task_mapping(episodes_jsonl, tasks_jsonl)

    # Step 2: Load the task mapping
    try:
        task_to_idx_map = load_task_mapping(tasks_jsonl)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Step 3: Find all chunk directories and process them
    chunk_dirs = [d for d in root_data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    if not chunk_dirs:
        print(f"Error: No 'chunk-*' directories found in '{root_data_dir}'.")
        print("Please ensure --data_dir points to the parent 'data' directory.")
        return

    for chunk_dir in chunk_dirs:
        print(f"\n--- Processing chunk: {chunk_dir.name} ---")
        process_dataset(chunk_dir, episodes_jsonl, task_to_idx_map)

    print("\nAll chunks have been processed successfully.")

if __name__ == "__main__":
    main()
