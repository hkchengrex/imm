N_KEYPOINTS=$1
python scripts/train.py --configs configs/paths/default.yaml configs/experiments/nabirds-"$1"pts.yaml