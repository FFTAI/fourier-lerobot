import argparse
import gc
import json
import shutil
from datetime import datetime, timezone
from html import parser
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                The first 3 columns represent the coordinates (x, y, z).
                The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)

    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000

    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)

    # Return sampled points
    return point_cloud[indices]


def create_colored_point_cloud_from_depth_oak(depth, far=1.0, near=0.1, num_points=10000):
    # color = cv2.resize(color, (960, 540), interpolation=cv2.INTER_LINEAR)

    # assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
    # Resize color image to match depth resolution

    # Create meshgrid for pixel coordinates
    # xmap = np.arange(color.shape[1])
    # ymap = np.arange(color.shape[0])
    fx = 570.687
    fy = 572.884
    cx = 633.181549072266
    cy = 348.350448608398

    xmap = np.arange(depth.shape[1])
    ymap = np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    # Calculate 3D coordinates
    points_z = depth  # / 0.001 #/ self.camera_info.scale
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    cloud = cloud.reshape([-1, 3])

    clip_low_x = -0.5
    clip_high_x = 0.5
    mask_x = (cloud[:, 0] > clip_low_x) & (cloud[:, 0] < clip_high_x)
    cloud = cloud[mask_x]

    # Clip points based on depth
    mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
    cloud = cloud[mask]
    # color = color.reshape([-1, 3])
    # color = color[mask]

    cloud = grid_sample_pcd(cloud, grid_size=0.005)

    if num_points > cloud.shape[0]:
        num_pad = num_points - cloud.shape[0]
        pad_points = np.zeros((num_pad, 3))
        cloud = np.concatenate([cloud, pad_points], axis=0)
    else:
        # Randomly sample points
        selected_idx = np.random.choice(cloud.shape[0], num_points, replace=True)
        cloud = cloud[selected_idx]

    # shuffle
    np.random.shuffle(cloud)
    return cloud


def match_timestamps(candidate, ref):
    closest_indices = []
    # candidate = np.sort(candidate)
    already_matched = set()
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        if idx > 0 and (
            idx == len(candidate) or np.fabs(t - candidate[idx - 1]) < np.fabs(t - candidate[idx])
        ):
            idx = idx - 1
        if idx not in already_matched:
            closest_indices.append(idx)
            already_matched.add(idx)
        else:
            print("Duplicate timestamp found: ", t, " trying to use next closest timestamp")
            if idx + 1 not in already_matched:
                closest_indices.append(idx + 1)
                already_matched.add(idx + 1)

    # print("closest_indices: ", len(closest_indices))
    return np.array(closest_indices)


def get_cameras(hdf5_path):
    # get camera keys
    image_path = hdf5_path.with_suffix("")
    # get all folder names in the image path
    camera_keys = [x.name for x in image_path.iterdir() if x.is_dir()]
    return camera_keys


def iso_to_datetime(filename: str) -> datetime:
    """
    Convert an ISO 8601 filename with microseconds back to a datetime object.
    Args:
        filename (str): The filename to parse, including or excluding the file extension.
    Returns:
        datetime: Parsed datetime object.
    """
    if "." in filename:
        base_name = filename.split(".")[0]  # Remove file extension
    else:
        base_name = filename
    return datetime.strptime(base_name, "%Y-%m-%dT%H-%M-%S_%f").replace(tzinfo=timezone.utc)


STATE_JOINT_NAMES = [
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_hip_pitch_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_pitch_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "head_yaw_joint",
    "head_roll_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
]

ACTION_JOINT_NAMES = STATE_JOINT_NAMES

STATE_HAND6DOF_NAMES = [
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
]
ACTION_HAND6DOF_NAMES = STATE_HAND6DOF_NAMES

STATE_POSE_NAMES = [
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
    "left_hand_ortho6d_1",
    "left_hand_ortho6d_2",
    "left_hand_ortho6d_3",
    "left_hand_ortho6d_4",
    "left_hand_ortho6d_5",
    "left_hand_ortho6d_6",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "right_hand_ortho6d_1",
    "right_hand_ortho6d_2",
    "right_hand_ortho6d_3",
    "right_hand_ortho6d_4",
    "right_hand_ortho6d_5",
    "right_hand_ortho6d_6",
    "head_x",
    "head_y",
    "head_z",
    "head_ortho6d_1",
    "head_ortho6d_2",
    "head_ortho6d_3",
    "head_ortho6d_4",
    "head_ortho6d_5",
    "head_ortho6d_6",
]


def filter_state_pose(state):
    return state[:18] + state[-6:]  # filter out the head xyz


ACTION_POSE_NAMES = [
    "left_hand_x",
    "left_hand_y",
    "left_hand_z",
    "left_hand_ortho6d_1",
    "left_hand_ortho6d_2",
    "left_hand_ortho6d_3",
    "left_hand_ortho6d_4",
    "left_hand_ortho6d_5",
    "left_hand_ortho6d_6",
    "right_hand_x",
    "right_hand_y",
    "right_hand_z",
    "right_hand_ortho6d_1",
    "right_hand_ortho6d_2",
    "right_hand_ortho6d_3",
    "right_hand_ortho6d_4",
    "right_hand_ortho6d_5",
    "right_hand_ortho6d_6",
    "head_ortho6d_1",
    "head_ortho6d_2",
    "head_ortho6d_3",
    "head_ortho6d_4",
    "head_ortho6d_5",
    "head_ortho6d_6",
]


def make_features(state_names, action_names, mode="video"):
    return {
        "observation.images.top": {
            "dtype": "video",
            "copy": True,
            "shape": (3, 800, 1280),
            "names": [
                "channel",
                "height",
                "width",
            ],
        },
        # "observation.depth.top": {
        #     "dtype": "depth",
        #     "shape": (1, 800, 1280),
        #     "names": [
        #         "channel",
        #         "height",
        #         "width",
        #     ],
        # },
        "observation.pointcloud": {  # TODO: make this env state?
            "dtype": "float32",
            "shape": ((4096 * 3),),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "observation.state.pose": {
            "dtype": "float32",
            "shape": (len(filter_state_pose(STATE_POSE_NAMES) + STATE_HAND6DOF_NAMES),),
            "names": filter_state_pose(STATE_POSE_NAMES) + STATE_HAND6DOF_NAMES,
        },
        "action.pose": {
            "dtype": "float32",
            "shape": (len(ACTION_POSE_NAMES + ACTION_HAND6DOF_NAMES),),
            "names": ACTION_POSE_NAMES + ACTION_HAND6DOF_NAMES,
        },
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main(
    raw_dir: Path,
    repo_id: str,
    task: str | None = None,
    mode: str = "video",
    pointcloud: bool = True,
    use_pose: bool = False,
    push_to_hub: bool = False,
    discard_frames=0,
):
    if mode not in {"video", "image"}:
        raise ValueError(mode)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    state_names = STATE_JOINT_NAMES[12:] + STATE_HAND6DOF_NAMES
    action_names = ACTION_JOINT_NAMES[12:] + ACTION_HAND6DOF_NAMES

    if use_pose:
        state_names = filter_state_pose(STATE_POSE_NAMES) + STATE_HAND6DOF_NAMES
        action_names = ACTION_POSE_NAMES + ACTION_HAND6DOF_NAMES

    FOURIER_FEATURES = make_features(state_names, action_names, mode=mode)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type="gr1t1",
        features=FOURIER_FEATURES,
        # tolerance_s=0.1,
        image_writer_threads=4,
        image_writer_processes=4,
    )

    hdf5_files = sorted(raw_dir.glob("*.hdf5"))
    num_episodes = len(hdf5_files)
    episodes = range(num_episodes)

    if task is None:
        metadata = json.load(open(raw_dir / "metadata.json"))
        metadata = {m["id"]: m for m in metadata}

    for ep_idx in episodes:
        ep_path = hdf5_files[ep_idx]

        with h5py.File(ep_path, "r") as ep:
            if use_pose:
                state = torch.from_numpy(
                    np.concatenate(
                        [ep["/state/pose"][:, :18], ep["/state/pose"][:, -6:], ep["/state/hand"][:]], axis=1
                    )
                )  # type: ignore
                # concatenate the robot state with the hand state
                action = torch.from_numpy(
                    np.concatenate([ep["/action/pose"][:], ep["/action/hand"][:]], axis=1)
                )  # type: ignore
            else:
                state = torch.from_numpy(
                    np.concatenate([ep["/state/robot"][:, 12:], ep["/state/hand"][:]], axis=1)
                )  # type: ignore
                # concatenate the ee pose state with the hand state
                action = torch.from_numpy(
                    np.concatenate([ep["/action/robot"][:, 12:], ep["/action/hand"][:]], axis=1)
                )  # type: ignore

                pose_state = torch.from_numpy(
                    np.concatenate(
                        [ep["/state/pose"][:, :18], ep["/state/pose"][:, -6:], ep["/state/hand"][:]], axis=1
                    )
                )  # type: ignore
                # concatenate the robot state with the hand state
                pose_action = torch.from_numpy(
                    np.concatenate([ep["/action/pose"][:], ep["/action/hand"][:]], axis=1)
                )  # type: ignore
            assert state.shape[1] == FOURIER_FEATURES["observation.state"]["shape"][0], (
                f"{state.shape=} {FOURIER_FEATURES['observation.state']['shape'][0]=}"
            )
            assert action.shape[1] == FOURIER_FEATURES["action"]["shape"][0], (
                f"{action.shape=} {FOURIER_FEATURES['action']['shape'][0]=}"
            )

            num_frames = None
            matched = None

            frames = {}

            for camera in get_cameras(ep_path.parent / ep_path.stem):
                vid_key = f"observation.images.{camera}"
                depth_key = f"observation.depth.{camera}"
                pc_key = "observation.pointcloud"  # TODO: multi

                raw_vid_dir = ep_path.parent / ep_path.stem / camera

                rgb_path = raw_vid_dir / "rgb.mp4"
                depth_path = raw_vid_dir / "depth.mkv"
                timestamps_path = raw_vid_dir / "timestamps.json"

                if mode == "video":
                    with open(timestamps_path, "r") as f:
                        timestamps = json.load(f)

                    image_timestamps = [iso_to_datetime(ts) for ts in timestamps]
                elif mode == "image":
                    image_timestamps = sorted(
                        [iso_to_datetime(ts.stem) for ts in raw_vid_dir.glob("rgb/*.png")]
                    )
                else:
                    raise ValueError(mode)

                data_ts = np.asarray(ep["/timestamp"])
                non_duplicate = np.where(np.diff(data_ts) > 0)[0]
                image_ts = np.asarray(
                    [ts.timestamp() for ts in image_timestamps if ts.timestamp() < data_ts[-1]]
                )

                matched = match_timestamps(data_ts[non_duplicate], image_ts)
                matched = matched[
                    discard_frames : -discard_frames if discard_frames else None
                ]  # remove the first and last 3 frames
                num_frames = len(matched)

                timestamps = list(image_ts)[discard_frames : -discard_frames if discard_frames else None]
                start_time = timestamps[0]
                timestamps = [(ts - start_time) for ts in timestamps]

                if mode == "video":
                    video_output_path = dataset.meta.get_video_file_path(ep_idx, vid_key)

                    (dataset.root / video_output_path).parent.mkdir(parents=True, exist_ok=True)
                    fname = dataset.root / video_output_path
                    if not fname.exists():
                        fname: Path = shutil.copy(rgb_path, dataset.root / video_output_path)

                        print(f"Copy video from {rgb_path} to {fname}")
                    else:
                        print(f"Already exists: {fname}")

                    # store the reference to the video frame
                    # frames[vid_key] = [
                    #     {"path": f"{fname.relative_to(dataset.root)}", "timestamp": t} for t in timestamps
                    # ]
                    frames[vid_key] = [
                        {"path": f"{fname.relative_to(dataset.root)}", "timestamp": i / dataset.fps}
                        for i in range(discard_frames, len(image_ts) - discard_frames)
                    ]
                elif mode == "image":
                    from tqdm import tqdm

                    image_output_path = dataset._get_image_file_path(
                        ep_idx, f"observation.images.{camera}", 0
                    ).parent

                    fname = dataset.root / image_output_path
                    fname.mkdir(parents=True, exist_ok=True)

                    print(f"Copying images from {raw_vid_dir} to {fname}")

                    images = sorted(raw_vid_dir.glob("rgb/*.png"))
                    print(f"Found {len(images)} images in {ep_path.with_suffix('')}")

                    frames[vid_key] = []
                    for i, frame_idx in enumerate(tqdm(range(discard_frames, len(images) - discard_frames))):
                        new_file_name = dataset._get_image_file_path(
                            ep_idx, f"observation.images.{camera}", frame_idx
                        )
                        shutil.copy(images[i], dataset.root / new_file_name)

                        frames[vid_key].append(
                            {
                                "path": new_file_name,
                                "timestamp": i / dataset.fps,
                            }
                        )

                if depth_path.exists():
                    # copy depth video
                    depth_output_path = dataset.meta.get_video_file_path(ep_idx, depth_key).with_suffix(
                        ".mkv"
                    )

                    (dataset.root / depth_output_path).parent.mkdir(parents=True, exist_ok=True)
                    fname = dataset.root / depth_output_path
                    if not fname.exists():
                        fname: Path = shutil.copy(depth_path, dataset.root / depth_output_path)

                        print(f"Copy video from {depth_path} to {fname}")
                    else:
                        print(f"Already exists: {fname}")

                    if pointcloud:
                        video_capture = cv2.VideoCapture(
                            filename=depth_path,
                            apiPreference=cv2.CAP_FFMPEG,
                            params=[
                                cv2.CAP_PROP_CONVERT_RGB,
                                0,  # false
                            ],
                        )
                        if not video_capture.isOpened():
                            print("Error reading video file")
                        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # depth_out_path = (
                        #     str(dataset.root) + "/" + str(video_output_path).replace("top", "depth")
                        # )
                        # parent_dir = os.path.dirname(depth_out_path)
                        # os.makedirs(parent_dir, exist_ok=True)
                        # video_writer = cv2.VideoWriter(
                        #     filename=depth_out_path,
                        #     fourcc=fourcc,
                        #     fps=fps,
                        #     frameSize=(1280, 800),  # 必须严格匹配 (width, height)
                        #     isColor=True,
                        # )
                        points = []
                        # depths = []
                        while True:
                            ret, frame = video_capture.read()
                            if ret:
                                # depths.append(torch.tensor(frame, dtype=torch.uint16).unsqueeze(0))
                                points.append(
                                    create_colored_point_cloud_from_depth_oak(frame * 1e-3, num_points=4096)
                                )
                                # frame = (frame / frame.max() * 256).astype(np.uint8)
                                # frame = cv2.applyColorMap(frame, cv2.COLORMAP_TURBO)
                                # video_writer.write(frame)
                            else:
                                break
                        # video_writer.release()
                        video_capture.release()

                        points = np.asarray(points)[
                            discard_frames : -discard_frames if discard_frames else None
                        ]
                        frames[pc_key] = torch.tensor(points, dtype=torch.float32).view(len(points), -1)
                        # print("point cloud shape", frames[pc_key].shape)
                        # depths = np.asarray(depths)[discard_frames:-discard_frames if discard_frames else None]
                        # frames[depth_key] = torch.tensor(depths, dtype=torch.uint16)
                        # print("depth shape", frames[depth_key].shape)
            if num_frames is None:
                raise ValueError("No frames found for episode")

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True
            frames["observation.state"] = state[matched]
            frames["action"] = action[matched]

            if not use_pose:
                frames["observation.state.pose"] = pose_state[matched]
                frames["action.pose"] = pose_action[matched]

            # frames["episode_index"] = torch.tensor([ep_idx] * num_frames)
            # frames["frame_index"] = torch.arange(0, num_frames, 1)

            # frames["timestamp"] = torch.from_numpy(np.asarray(ep["/timestamp"])[matched] - start_time)
            # frames["timestamp"] = torch.from_numpy(timestamps)
            frames["next.done"] = done

            for i in range(num_frames):
                frame = {k: v[i] for k, v in frames.items()}
                dataset.add_frame(frame)
        dataset.save_episode(
            task=task if task else metadata[ep_path.stem].get("prompt", ""), encode_videos=False
        )

        gc.collect()

    dataset.consolidate(run_compute_stats=True, keep_image_files=False)

    if push_to_hub:
        dataset.push_to_hub(private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--mode", type=str, default="video")
    parser.add_argument("--push-to-hub", action="store_true")

    parser.add_argument("--pose", action="store_true", default=False)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    repo_id = args.repo_id

    print(f"Pushing to hub: {args.push_to_hub}")

    main(
        raw_dir,
        repo_id=repo_id,
        task=args.task,
        mode=args.mode,
        push_to_hub=args.push_to_hub,
        use_pose=args.pose,
        pointcloud=True,
    )
