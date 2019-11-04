import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='tf-pose-estimation run by folder')
parser.add_argument('--folder', type=str, default='/Users/tomhata/PycharmProjects/mocap2camera/data/raw/actigraphy-take-1/')
parser.add_argument('--model', type=str, default='cmu',
                    help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')


args = parser.parse_args()

video_names = [f for f in os.listdir(args.folder) if not f.startswith(".")]

for video in video_names:
    cli_call = [
            "python",
            "run_video.py",
            "--video",
            os.path.join(args.folder, video),
            "--model",
            args.model
        ]
    print(" ".join(cli_call))
    subprocess.call(cli_call)
