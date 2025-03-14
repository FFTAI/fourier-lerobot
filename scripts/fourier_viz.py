

import warnings
import argparse
from pathlib import Path

import cv2
import h5py
from tqdm.rich import tqdm
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation as R
import numpy as np
from loguru import logger
from datetime import datetime, timezone

import json
from pathlib import Path

from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# check if system is windows
import platform
if platform.system() == "Windows":
    print("Windows system detected, using `cv2.imdecode` for reading images because Chinese characters in path may cause problems. Fuck Windows.")
    def rgb_imread(path: Path):
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        return image
    
    def depth_imread(path: Path):
        buf = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        return image
    
else:
    def rgb_imread(path: Path):
        return cv2.imread(str(path))[:, :, ::-1]
    
    def depth_imread(path: Path):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        return image

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
    return datetime.strptime(base_name, "%Y-%m-%dT%H-%M-%S_%f").replace(
        tzinfo=timezone.utc
    )
    
def ortho6d_to_so3(ortho6d):
    """
    Convert from continuous 6D rotation representation to SO(3), adapted from
    On the Continuity of Rotation Representations in Neural Networks
    https://arxiv.org/pdf/1812.07035.pdf
    https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
    """
    x_raw = ortho6d[:3]
    y_raw = ortho6d[3:6]

    x = x_raw / np.linalg.norm(x_raw)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def ortho6d_to_R(ortho6d):
    return R.from_matrix(ortho6d_to_so3(ortho6d))


fx = 561.6825561523438
fy = 561.4020385742188

cx = 642.8319702148438
cy = 388.20196533203125

JOINT_NAMES = [
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

HAND_JOINT_NAMES = [
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


def make_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    name="view",
                    contents=[
                        rrb.Spatial3DView(
                            origin="world",
                            contents=["world/camera/top/**", "world/robot/**"],
                        ),
                        rrb.Spatial2DView(
                            origin="/world/camera/top/rgb/image",
                            name="rgb",
                        ),
                        rrb.Spatial2DView(
                            origin="/world/camera/top/depth/image",
                            name="depth",
                        ),
                    ],
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"left_joints/left_{joint}",
                            origin=f"left_joints/left_{joint}",
                        )
                        for joint in [
                            "shoulder_pitch_joint",
                            "shoulder_roll_joint",
                            "shoulder_yaw_joint",
                            "elbow_pitch_joint",
                            "wrist_yaw_joint",
                            "wrist_roll_joint",
                            "wrist_pitch_joint",
                        ]
                    ],
                    name="left_joints",
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"right_joints/right_{joint}",
                            origin=f"right_joints/right_{joint}",
                        )
                        for joint in [
                            "shoulder_pitch_joint",
                            "shoulder_roll_joint",
                            "shoulder_yaw_joint",
                            "elbow_pitch_joint",
                            "wrist_yaw_joint",
                            "wrist_roll_joint",
                            "wrist_pitch_joint",
                        ]
                    ],
                    name="right_joints",
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"left_hand/L_{joint}",
                            origin=f"left_hand/L_{joint}",
                        )
                        for joint in [
                            "pinky_proximal_joint",
                            "ring_proximal_joint",
                            "middle_proximal_joint",
                            "index_proximal_joint",
                            "thumb_proximal_pitch_joint",
                            "thumb_proximal_yaw_joint",
                        ]
                    ],
                    name="left_hand",
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name=f"right_hand/R_{joint}",
                            origin=f"right_hand/R_{joint}",
                        )
                        for joint in [
                            "pinky_proximal_joint",
                            "ring_proximal_joint",
                            "middle_proximal_joint",
                            "index_proximal_joint",
                            "thumb_proximal_pitch_joint",
                            "thumb_proximal_yaw_joint",
                        ]
                    ],
                    name="right_hand",
                ),
            ]
        ),
        collapse_panels=True,
    )
    # rr.send_blueprint(blueprint)
    # logger.info("Blueprint sent.")
    return blueprint


def log_episode(root_folder: Path, episode: str):
    logger.info(f"Loading episode {episode}")
    
    # find the hdf5 file
    try:
        episode_folder = root_folder.rglob(f"{episode}.hdf5").__next__().parent
    except StopIteration:
        logger.error(f"Episode {episode} not found")
        raise FileNotFoundError(f"Episode {episode} not found")
    rr.log("/world", rr.ViewCoordinates.FLU, static=True)
    rr.log(
        "world/camera/top",
        rr.Transform3D(
            translation=[0.11252, 0, 0.55029],
            mat3x3=R.from_euler("XYZ", [0, 44.5, 0], degrees=True).as_matrix(),
            from_parent=False,
        ),
        static=True,
    )
    rr.log(
        "world/camera/top/rgb/image",
        rr.Pinhole(
            resolution=[1280, 800],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            camera_xyz=rr.ViewCoordinates.FLU,
            image_plane_distance=1.0,
        ),
        static=True,
    )

    rr.log(
        "world/camera/top/depth/image",
        rr.Pinhole(
            resolution=[1280, 800],
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            camera_xyz=rr.ViewCoordinates.FLU,
        ),
        static=True,
    )

    import os

    # def log_rgb(executor=None):

    #     import os

    def log_rgb(executor=None):
        # Locate the MP4 file and timestamps.json
        video_path = next((root_folder / f"{episode}" / "top").glob("*.mp4"), None)
        if video_path is None:
            print("Error: No video found in the specified directory.")
            return

        timestamp_file = Path(video_path.parent) / "timestamps.json"

        # Load timestamps
        with open(timestamp_file, "r", encoding="utf-8") as file:
            timestamps = json.load(file)

        total_frames = len(timestamps)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
 
        # Process each frame with tqdm progress bar
        for i in tqdm(range(total_frames), desc="Logging RGB images", total=total_frames):
            def _log_image(i):
                rr.set_time_seconds("0timestamp", iso_to_datetime(timestamps[i]).timestamp())
                rr.set_time_sequence("image_index", i)

                # Move to the i-th frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, image = cap.read()
                if not ret or image is None:
                    return


                if len(image.shape) == 3:
                    image = image[:, :, 0]  # Convert to grayscale

                rr.log(f"world/camera/top/rgb/image", rr.Image(image))

            if executor:
                executor.submit(_log_image, i)
            else:
                _log_image(i)

        cap.release()  # Close video file after all frames are logged


    def log_depth(executor=None):
        # Locate the MP4 file and timestamps.json
        video_path = next((root_folder / f"{episode}" / "top").glob("*.mkv"), None)
        if video_path is None:
            print("Error: No video found in the specified directory.")
            return

        timestamp_file = Path(video_path.parent) / "timestamps.json"

        # Load timestamps
        with open(timestamp_file, "r", encoding="utf-8") as file:
            timestamps = json.load(file)

        total_frames = len(timestamps)

        # Open video
        cap = cv2.VideoCapture(
                    filename=video_path,
                    apiPreference=cv2.CAP_FFMPEG,
                    params=[
                        cv2.CAP_PROP_CONVERT_RGB,
                        0,  # false
                    ],
                )
 
        # Process each frame with tqdm progress bar
        for i in tqdm(range(total_frames), desc="Logging depth images", total=total_frames):
            def _log_image(i):
                rr.set_time_seconds("0timestamp", iso_to_datetime(timestamps[i]).timestamp())
                rr.set_time_sequence("image_index", i)

                # Move to the i-th frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, image = cap.read()

                if not ret or image is None:
                    return


                if len(image.shape) == 3:
                    image = image[:, :, 0]  # Convert to grayscale

                rr.log(f"world/camera/top/depth/image", rr.DepthImage(image))

            if executor:
                executor.submit(_log_image, i)
            else:
                _log_image(i)

        cap.release()  # Close video file after all frames are logged

                
    def log_data():
        with h5py.File(episode_folder / f"{episode}.hdf5", "r") as f:
            timestamps = f["timestamp"][:]
            
            for i, ts in enumerate(tqdm(timestamps, desc="Loading data")):
                rr.set_time_seconds("0timestamp", ts)
                rr.set_time_sequence("state_index", i)

                rr.log(
                    "world/robot/end_effectors",
                    rr.Boxes3D(
                        centers=[f["state/pose"][i][:3], f["state/pose"][i][9:12]],
                        quaternions=[
                            ortho6d_to_R(f["state/pose"][i][3:9]).as_quat(),
                            ortho6d_to_R(f["state/pose"][i][12 : 12 + 6]).as_quat(),
                        ],
                        sizes=[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                    ),
                )

                for j in range(18, 18 + 7):
                    rr.log(
                        f"left_joints/{JOINT_NAMES[j]}/action",
                        rr.Scalar(f["action/robot"][i][j]),
                    )
                    rr.log(
                        f"left_joints/{JOINT_NAMES[j]}/state",
                        rr.Scalar(f["state/robot"][i][j]),
                    )

                for j in range(18 + 7, 18 + 7 + 7):
                    rr.log(
                        f"right_joints/{JOINT_NAMES[j]}/action",
                        rr.Scalar(f["action/robot"][i][j]),
                    )
                    rr.log(
                        f"right_joints/{JOINT_NAMES[j]}/state",
                        rr.Scalar(f["state/robot"][i][j]),
                    )

                for j in range(len(f["action/hand"][i])):
                    side = "left" if HAND_JOINT_NAMES[j].startswith("L_") else "right"
                    rr.log(
                        f"{side}_hand/{HAND_JOINT_NAMES[j]}/action",
                        rr.Scalar(f["action/hand"][i][j]),
                    )

                    rr.log(
                        f"{side}_hand/{HAND_JOINT_NAMES[j]}/state",
                        rr.Scalar(f["state/hand"][i][j]),
                    )

    log_data()
    log_rgb()
    log_depth()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", "-f", type=str, required=True, help="The root folder of the dataset"
    )
    parser.add_argument(
        "--episode", "-e", type=str, required=True, help="The episode to visualize"
    )
    parser.add_argument("--rr-host", "-r", type=str, default=None, help="The rerun host")
    args = parser.parse_args()

    root_folder = Path(args.folder)
    
    try:
        root_folder = root_folder.rglob(f"{args.episode}.hdf5").__next__().parent
    except StopIteration:
        logger.error(f"Episode {args.episode} not found")
        raise FileNotFoundError(f"Episode {args.episode} not found")
    
    blueprint = make_blueprint()

    if args.rr_host:
        rr.init("fourier_data_viewer", spawn=False, default_blueprint=blueprint)
        if args.rr_host == "web":
            rr.serve_web(open_browser=False, default_blueprint=blueprint)
            # rr.connect_tcp("127.0.0.1:9877")
        else:
            rr.connect_tcp(args.rr_hos, default_blueprint=blueprint)
    else:
        rr.init("fourier_data_viewer", spawn=True, default_blueprint=blueprint)
        
    # rr.send_blueprint(blueprint)
        

    
    try:
        # if args.episode == "all":
        #     for episode_folder in sorted(root_folder.glob("*.hdf5")):
        #         log_episode(root_folder, episode_folder.stem)
        # else:
        log_episode(root_folder, args.episode)
        
        logger.info(f"Done logging episode {args.episode}")
        
        if args.rr_host is None or args.rr_host == "web":
            import time
            while True:
                time.sleep(10)
    except KeyboardInterrupt or FileNotFoundError:
        
        rr.disconnect()
        import os

        os._exit(1)
        
    

    