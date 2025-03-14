# Fourier ActionNet Dataset

## Data Introduction
The data collected from two primary sources: the **robot side** and the **camera side**. The HDF5 file contains the robot-side data, while the camera-side data is stored in the corresponding episode folder. While there is also a **metadata.json** file in the dataset, which contains all episodes id, and it's prompt.

## Download the dataset
First, you can easily download the dataset online, which will be in a .tar file. After downloading, use the following command to extract all files:

```bash
find . -type f -name "*.tar" | xargs -I {} tar -xf {}
```

After untar, the dataset will be in the following structure

```txt
├── 01JH00FCRH6EIBDXTA # episode id
│   └── top
│       ├── depth.mkv # z16 depth video encoded in mkv format
│       ├── rgb.mp4 # h264 rgb video
│       └── timestamps.json
├── 01JH00FCRH6EIBDXTA.hdf5
├── 01JH00FRJ5YISEASEL
│   └── top
│       ├── depth.mkv
│       ├── rgb.mp4
│       └── timestamps.json
├── 01JH00FRJ5YISEASEL.hdf5
├── Metadata.json # metadata of the task, including prompt and all episodes id of the task
...
```

## Data Viewer
We provide a simple data viewer to visualize the data. 
First, you need to install the dependencies, there are two ways to do it:

```bash
# Install dependencies
pip install -r requirements.txt
# Install with optional dependencies in the main package
pip install -e .[fourier_viz]
```

With the dependencies installed, you can launch the data viewer. You can use the following command to launch the viewer:

```bash
# Usage with help message:
python scripts/fourier_viz.py -h

# Example usage:
python scripts/fourier_viz.py -d DATASET_PATH -e EPISODE_ID
```


## Data Preparation
Since our training pipeline relies on the LeRobotDatasetV2 format, you can use the following command to easily convert your dataset to the LeRobotDatasetV2 format. We would be delighted if our dataset could help accelerate the exciting era of humanoids.

```bash
python scripts/convert_to_lerobot_v2.py --raw-dir DATASET_PATH --repo-id FourierIntelligence/ActionNet
```


### Data Structure
The robot-side data is stored in an HDF5 file for each episode. Below is the structure of the data stored in the HDF5 file for one episode. For more detailed information, please refer to the [Data Explanation](#data-explanation).


#### HDF5 file structure
 ```txt
01JH00FCRH6EIBDXTA.hdf5
├── action # Robot action data
│   ├── hand [12,x] or [24,x] # Dexhand data with Fourier hand
│   ├── pose [27,x] # End link data
│   └── robot [32 or 29] # All joint data in humanoids.
├── state # Robot state data
│   ├── hand [12,x] or [24,x]
│   ├── pose [27,x]
│   └── robot [32 or 29] 
├── timestamp # Timestamp for both state and action
└── attributes # HDF5 attributes
```

#### Metadata.json
```txt
Metadata.json
{
    {
        "id": "01JH00FCRH6EIBDXTA",
        "prompt": "Pick the lemon and put it in the box.",
    }
    ...
}
```

### Data Explanation
#### Robot-side data(hdf5)
The robot-side data is organized into three main categories: **robot state**, **robot action**, and **timestamp**. Each of these categories contains specific types of data related to the robot’s operation. Notably, both the **robot state** and **robot action** data are stored using similar classes. Below are the classes and their details:

- **Hand Data:** The data contains either (12, x) or (24, x) entries. For the Fourier hand with 6 DOF, it consists of 12 data points (6 for the left hand and 6 for the right hand). For the 12 DOF hand, it consists of 24 data points (12 for each hand).
- **Pose Data:** The data has dimensions (27, x). This includes position and gesture data for the end-link of both arms and the head. In our dataset, the **first two columns of the rotational matrix** represent gesture data, while the **position vector** represents position data. Therefore, each end link (left arm, right arm, head) has 9 data points.
- **Robot Data**: This data contains the joint position information of the robot. For the **GR1-T1** and **GR1-T2** robots, there are 32 data points. For the **GR2** robot, there are 29 data points, covering all joint positions of the robot.

#### Camera-side data(rgb & depth)
Data from the camera side is stored in the folder corresponding to each episode. The data is stored in the following format:
- **rgb.mp4:** This file contains the RGB video encoded in H264 format.
- **depth.mkv:** This file contains the depth video encoded in Z16 format.
- **timestamps.json:** This JSON file contains timestamps for each frame in the videos, providing synchronization between the camera and robot data.

> Note: All state and action data has already been aligned based on the timestamps in a same hdf5, so they are sharing one timestamp array.


