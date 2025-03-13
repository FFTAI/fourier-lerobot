# Fourier ActionNet Dataset


### Download the dataset

After download, use the following command to untar all files

```bash
find . -type f -name "*.tar" | xargs -I {} tar -xf {}
```

After untar, the dataset will be in the following structure

```txt
├── 01JH00FCRH6EIBDXTA
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
...
```

Then you can use the following command to convert the dataset to `LeRobotDatasetV2` format

```bash
python scripts/convert_to_lerobot_v2.py --raw-dir DATASET_PATH --repo-id FourierIntelligence/ActionNet
```

