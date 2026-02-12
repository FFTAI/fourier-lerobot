import gc
import json
import shutil
import os
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import imageio.v3 as iio

import argparse
import cv2
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LEROBOT_HOME


@dataclass
class JointFieldConfig:
    state_joint_names: List[str]
    action_joint_names: List[str]
    hand_joint_names: List[str]
    pose_names: List[str]


@dataclass
class BaseFieldConfig:
    state_base_names: List[str]
    action_base_names: List[str]


@dataclass
class CameraConfig:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    scale: float = 1.0


@dataclass 
class RobotConfig:
    name: str
    joints: JointFieldConfig
    cameras: Dict[str, CameraConfig]
    features: Dict[str, Any]
    base: Optional[BaseFieldConfig] = None


GR3QNEXO_CONFIG = RobotConfig(
    name="gr3qnexo",
    joints=JointFieldConfig(
        state_joint_names=[
            "head_yaw_joint", "head_pitch_joint", "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ],
        action_joint_names=[
            "head_yaw_joint", "head_pitch_joint", "waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint",
            "left_wrist_yaw_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint",
            "right_wrist_yaw_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_pinky_proximal_joint", "L_ring_proximal_joint", "L_middle_proximal_joint", "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint", "L_thumb_proximal_yaw_joint",
            "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_middle_proximal_joint", "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint", "R_thumb_proximal_yaw_joint",
        ],
        pose_names=[
            "left_hand_x", "left_hand_y", "left_hand_z",
            "left_hand_ortho6d_1", "left_hand_ortho6d_2", "left_hand_ortho6d_3", "left_hand_ortho6d_4", "left_hand_ortho6d_5", "left_hand_ortho6d_6",
            "right_hand_x", "right_hand_y", "right_hand_z",
            "right_hand_ortho6d_1", "right_hand_ortho6d_2", "right_hand_ortho6d_3", "right_hand_ortho6d_4", "right_hand_ortho6d_5", "right_hand_ortho6d_6",
            "head_x", "head_y", "head_z",
            "head_ortho6d_1", "head_ortho6d_2", "head_ortho6d_3", "head_ortho6d_4", "head_ortho6d_5", "head_ortho6d_6",
        ],
    ),
    base=BaseFieldConfig(
        state_base_names=["base_height", "base_pitch"],
        action_base_names=["vel_height", "vel_pitch", "base_yaw", "vel_x", "vel_y", "vel_yaw"],
    ),
    cameras={
        "oak": CameraConfig(width=1280, height=800, fx=570.687, fy=572.884, cx=633.181549072266, cy=348.350448608398, scale=1.0)
    },
    features={"video_shape": [1280, 800], "fps": 30},
)


def match_timestamps(candidate, ref):
    closest_indices = []
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx - 1]) < np.fabs(t - candidate[idx])):
            idx = idx - 1
        closest_indices.append(idx)
    return np.array(closest_indices)


def generate_timestamps(start_time: datetime, end_time: datetime, delta_ms: int = 33) -> List[datetime]:
    if start_time > end_time:
        raise ValueError("start_time must be <= end_time")
    timestamps = []
    current_time = start_time
    delta = timedelta(milliseconds=delta_ms)
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += delta
    return timestamps


def sample_stat_by_timestamps(stat: torch.Tensor, candidate_timestamps: pd.Series, ref_timestamps) -> torch.Tensor:
    candidate_numeric = candidate_timestamps.astype('int64') / 1e9
    if not isinstance(ref_timestamps, np.ndarray):
        ref_numeric = np.array([dt.timestamp() for dt in ref_timestamps])
    else:
        ref_numeric = ref_timestamps
    matched_indices = match_timestamps(candidate_numeric.values, ref_numeric)
    return stat[matched_indices]


def filter_state_pose(pose_names: List[str]) -> List[str]:
    return pose_names[:18] + pose_names[-6:]


def filter_action_pose(pose_names: List[str]) -> List[str]:
    return pose_names[:18] + pose_names[-6:]


def get_state_names(joints: JointFieldConfig, base: BaseFieldConfig, use_pose: bool = False) -> List[str]:
    if use_pose:
        return filter_state_pose(joints.pose_names) + joints.hand_joint_names
    return joints.state_joint_names + joints.hand_joint_names + base.state_base_names


def get_action_names(joints: JointFieldConfig, base: BaseFieldConfig, use_pose: bool = False) -> List[str]:
    if use_pose:
        return filter_action_pose(joints.pose_names) + joints.hand_joint_names
    return joints.action_joint_names + joints.hand_joint_names + base.action_base_names


def get_state_joint_names(joints: JointFieldConfig) -> List[str]:
    return joints.state_joint_names + joints.hand_joint_names


def get_state_base_names(base: BaseFieldConfig) -> List[str]:
    return base.state_base_names if base else []


def get_action_joint_names(joints: JointFieldConfig) -> List[str]:
    return joints.action_joint_names + joints.hand_joint_names


def get_action_base_names(base: BaseFieldConfig) -> List[str]:
    return base.action_base_names if base else []


def make_features(robot_config: RobotConfig, use_pose: bool = False, mode: str = "video",
                  pointcloud: bool = False, video_shape: tuple = (1280, 800), camera: List[str] = ["top"]) -> dict:
    joints = robot_config.joints
    base = robot_config.base
    state_names = get_state_names(joints, base, use_pose)
    action_names = get_action_names(joints, base, use_pose)

    features = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        "observation.state.pose": {
            "dtype": "float32",
            "shape": (len(filter_state_pose(joints.pose_names) + joints.hand_joint_names),),
            "names": filter_state_pose(joints.pose_names) + joints.hand_joint_names,
        },
        "action.pose": {
            "dtype": "float32",
            "shape": (len(filter_action_pose(joints.pose_names) + joints.hand_joint_names),),
            "names": filter_action_pose(joints.pose_names) + joints.hand_joint_names,
        },
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    for cam in camera:
        features[f"observation.images.{cam}"] = {
            "dtype": "video", "shape": (3, video_shape[1], video_shape[0]), "names": ["channel", "height", "width"],
        }

    if pointcloud:
        features["observation.pointcloud"] = {"dtype": "float32", "shape": ((4096 * 3),)}

    return features


class StateConverter:
    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.state_joint_names = get_state_joint_names(robot_config.joints)
        self.state_base_names = get_state_base_names(robot_config.base)

    def _parse_states(self, state_series: pd.Series) -> torch.Tensor:
        num_frames = len(state_series)
        num_states = len(self.state_joint_names)
        first_sample = state_series.iloc[0]
        sample_state_names = [state['name'] for state in first_sample]

        reorder_indices = []
        for name in self.state_joint_names:
            reorder_indices.append(sample_state_names.index(name) if name in sample_state_names else -1)

        def extract_positions(joint_array):
            return np.array([joint['value'] for joint in joint_array])

        all_positions = np.array(state_series.apply(extract_positions).tolist())
        result = np.zeros((num_frames, num_states), dtype=np.float32)
        for i, idx in enumerate(reorder_indices):
            if idx != -1:
                result[:, i] = all_positions[:, idx]
        return torch.from_numpy(result).float()

    def _parse_base_states(self, base_state_series: pd.Series) -> torch.Tensor:
        num_frames = len(base_state_series)
        num_states = len(self.state_base_names)
        result = np.zeros((num_frames, num_states), dtype=np.float32)
        for i, row in base_state_series.items():
            stand_pose = row[0]['stand_pose']
            result[i, 0] = stand_pose[0]
            result[i, 1] = stand_pose[1]
        return torch.from_numpy(result).float()

    def _get_timestamps(self, file_path: Path) -> pd.Series:
        return pd.read_parquet(file_path)["timestamp_utc"]

    def convert_state_base(self, file_path: Path) -> torch.Tensor:
        return self._parse_base_states(pd.read_parquet(file_path)["observation.base_state"])

    def convert_state(self, file_path: Path) -> torch.Tensor:
        return self._parse_states(pd.read_parquet(file_path)["observation.state"])


class ActionConverter:
    def __init__(self, robot_config: RobotConfig):
        self.robot_config = robot_config
        self.action_joint_names = get_action_joint_names(robot_config.joints)
        self.action_base_names = get_action_base_names(robot_config.base)

    def _parse_actions(self, action_series: pd.Series) -> torch.Tensor:
        num_frames = len(action_series)
        num_actions = len(self.action_joint_names)
        first_sample = action_series.iloc[0]
        sample_action_names = [action['name'] for action in first_sample]

        reorder_indices = []
        for name in self.action_joint_names:
            reorder_indices.append(sample_action_names.index(name) if name in sample_action_names else -1)

        def extract_positions(joint_array):
            return np.array([joint['value'] for joint in joint_array])

        all_positions = np.array(action_series.apply(extract_positions).tolist())
        result = np.zeros((num_frames, num_actions), dtype=np.float32)
        for i, idx in enumerate(reorder_indices):
            if idx != -1:
                result[:, i] = all_positions[:, idx]
        return torch.from_numpy(result).float()

    def _parse_base_actions(self, base_action_series: pd.Series) -> torch.Tensor:
        num_frames = len(base_action_series)
        num_actions = len(self.action_base_names)
        result = np.zeros((num_frames, num_actions), dtype=np.float32)

        for i, row in enumerate(base_action_series):
            name_to_position = {action['name']: action['value'] for action in row}
            for j, action_name in enumerate(self.action_base_names):
                if action_name in name_to_position:
                    if action_name in ["vel_pitch", "vel_height"]:
                        result[i, j] = name_to_position[action_name] if i == 0 else result[i-1, j] + name_to_position[action_name]
                    else:
                        result[i, j] = name_to_position[action_name]
        return torch.from_numpy(result).float()

    def _get_timestamps(self, file_path: Path) -> pd.Series:
        return pd.read_parquet(file_path)["timestamp_utc"]

    def convert_action(self, file_path: Path) -> torch.Tensor:
        return self._parse_actions(pd.read_parquet(file_path)["action"])

    def convert_action_base(self, file_path: Path) -> torch.Tensor:
        return self._parse_base_actions(pd.read_parquet(file_path)["action.base"])


class VideoConverter:
    def __init__(self, video_shape: tuple = (480, 480), square_crop: bool = False):
        self.video_shape = video_shape
        self.square_crop = square_crop

    def _parse_video(self, video_series: pd.Series, cam: str) -> torch.Tensor:
        image_key = f"observation.images.camera_{cam}"
        frames = np.zeros((len(video_series), self.video_shape[1], self.video_shape[0], 3), dtype=np.uint8)
        for i, frame in enumerate(video_series[image_key]):
            img = iio.imread(bytes(frame), extension='.avif')
            if self.square_crop:
                h, w, _ = img.shape
                min_dim = min(h, w)
                start_h, start_w = (h - min_dim) // 2, (w - min_dim) // 2
                img = img[start_h:start_h + min_dim, start_w:start_w + min_dim]
                img = cv2.resize(img, self.video_shape)
            else:
                img = cv2.resize(img, self.video_shape)
            frames[i] = img
        return torch.from_numpy(frames)

    def _get_timestamps(self, file_path: Path) -> pd.Series:
        return pd.read_parquet(file_path)["timestamp_utc"]

    def convert_video(self, file_path: Path) -> torch.Tensor:
        cam = file_path.stem.split(".")[-1].split("_")[-1]
        video_df = pd.read_parquet(file_path)
        image_key = f"observation.images.camera_{cam}"
        video_series = video_df[[image_key, "timestamp_utc"]]
        return self._parse_video(video_series, cam)


class DataConverter:
    def __init__(self, robot_config: RobotConfig, video_shape: tuple = (480, 480), square_crop: bool = False):
        self.robot_config = robot_config
        self.state_converter = StateConverter(robot_config)
        self.action_converter = ActionConverter(robot_config)
        self.video_converter = VideoConverter(video_shape, square_crop)
        self.timestamps = None

    def generate_timestamps(self, ep_path: Path):
        video_ts = self.video_converter._get_timestamps(ep_path / "observation.images.camera_top.parquet")
        action_ts = self.action_converter._get_timestamps(ep_path / "action.parquet")
        action_base_ts = self.action_converter._get_timestamps(ep_path / "action.base.parquet")
        state_ts = self.state_converter._get_timestamps(ep_path / "observation.state.parquet")
        state_base_ts = self.state_converter._get_timestamps(ep_path / "observation.base_state.parquet")

        start_time = max(video_ts.min(), action_ts.min(), action_base_ts.min(), state_ts.min(), state_base_ts.min())
        end_time = min(video_ts.max(), action_ts.max(), action_base_ts.max(), state_ts.max(), state_base_ts.max())
        timestamps = generate_timestamps(start_time, end_time)
        self.timestamps = np.array([dt.timestamp() for dt in timestamps])

    def match_ts(self, stat: torch.Tensor, stat_timestamps: pd.Series) -> torch.Tensor:
        return sample_stat_by_timestamps(stat, stat_timestamps, self.timestamps)

    def convert_action(self, file_path: Path) -> torch.Tensor:
        actions = self.action_converter.convert_action(file_path)
        action_ts = self.action_converter._get_timestamps(file_path)
        sampled = self.match_ts(actions, action_ts)

        base_file = file_path.parent / "action.base.parquet"
        if base_file.exists():
            base_actions = self.action_converter.convert_action_base(base_file)
            base_ts = self.action_converter._get_timestamps(base_file)
            sampled_base = self.match_ts(base_actions, base_ts)
            min_t = min(sampled.shape[0], sampled_base.shape[0])
            sampled = torch.cat((sampled[:min_t], sampled_base[:min_t]), dim=1)
        return sampled

    def convert_state(self, file_path: Path) -> torch.Tensor:
        states = self.state_converter.convert_state(file_path)
        state_ts = self.state_converter._get_timestamps(file_path)
        sampled = self.match_ts(states, state_ts)

        base_file = file_path.parent / "observation.base_state.parquet"
        if base_file.exists():
            base_states = self.state_converter.convert_state_base(base_file)
            base_ts = self.state_converter._get_timestamps(base_file)
            sampled_base = self.match_ts(base_states, base_ts)
            min_t = min(sampled.shape[0], sampled_base.shape[0])
            sampled = torch.cat((sampled[:min_t], sampled_base[:min_t]), dim=1)
        return sampled

    def convert_video(self, file_path: Path) -> torch.Tensor:
        video = self.video_converter.convert_video(file_path)
        video_ts = self.video_converter._get_timestamps(file_path)
        return self.match_ts(video, video_ts)


def convert(
    raw_dir: Path,
    repo_id: str,
    robot_config: RobotConfig = GR3QNEXO_CONFIG,
    task: str | None = None,
    mode: str = "video",
    video_config: str | None = "480x480",
    square_crop: bool = False,
    pointcloud: bool = False,
    use_pose: bool = False,
    camera: list = ["top"],
):
    video_shape = (480, 480) if video_config is None else tuple(map(int, video_config.split("x")))
    converter = DataConverter(robot_config, video_shape, square_crop)

    features = make_features(robot_config, use_pose=use_pose, mode=mode, pointcloud=pointcloud, video_shape=video_shape, camera=camera)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    dataset = LeRobotDataset.create(
        repo_id=repo_id, fps=30, robot_type=robot_config.name, features=features,
        image_writer_threads=1, image_writer_processes=1,
    )

    episode_folders = sorted([p for p in raw_dir.rglob("episode*") if p.is_dir()])
    num_episodes = len(episode_folders)

    for ep_idx, ep_path in enumerate(episode_folders):
        print(f"Processing episode {ep_idx+1}/{num_episodes}: {ep_path.name}")
        metadata = json.load(open(ep_path / "metadata.json"))
        converter.generate_timestamps(ep_path)

        state_file = ep_path / "observation.state.parquet"
        action_file = ep_path / "action.parquet"
        if not state_file.exists() or not action_file.exists():
            print(f"Missing files for episode {ep_idx}, skipping...")
            continue

        episode_states = converter.convert_state(state_file)
        episode_actions = converter.convert_action(action_file)

        frames = {}
        for cam in camera:
            video_file = ep_path / f"observation.images.camera_{cam}.parquet"
            frames[f"observation.images.{cam}"] = converter.convert_video(video_file)

        num_frames = episode_states.shape[0]
        if not use_pose:
            frames["observation.state.pose"] = torch.zeros((num_frames, *features["observation.state.pose"]["shape"]))
            frames["action.pose"] = torch.zeros((num_frames, *features["action.pose"]["shape"]))

        frames["observation.state"] = episode_states
        frames["action"] = episode_actions
        ep_task = task if task else metadata.get("notes", " ")

        for i in range(num_frames):
            frame = {k: v[i] for k, v in frames.items()}
            frame["next.done"] = np.array([False])

            if i == num_frames - 6:
                frame["next.done"] = np.array([True])
                dataset.add_frame(frame)
                break
            dataset.add_frame(frame)

        dataset.save_episode(task=ep_task, encode_videos=True)

    dataset.consolidate(run_compute_stats=True, keep_image_files=False)
    print(f"Dataset saved to {LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument("--video-config", type=str, default="480x480")
    parser.add_argument("--square-crop", action="store_true", default=False)
    parser.add_argument("--pointcloud", action="store_true", default=False)
    parser.add_argument("--pose", action="store_true", default=False)
    parser.add_argument("--camera", type=str, nargs="+", default=["top"])
    args = parser.parse_args()

    convert(
        raw_dir=Path(args.raw_dir),
        repo_id=args.repo_id,
        robot_config=GR3QNEXO_CONFIG,
        task=args.task,
        mode=args.mode,
        video_config=args.video_config,
        square_crop=args.square_crop,
        pointcloud=args.pointcloud,
        use_pose=args.pose,
        camera=args.camera,
    )
