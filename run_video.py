import argparse
from datetime import datetime
import json
import logging
import os
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.json_tools import humans_to_keypoints_dict

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--log', type=str, required=False, default=None)
    parser.add_argument('--output', type=str, required=False, default=None)
    args = parser.parse_args()

    file_name = os.path.basename(args.video)
    file_prefix = os.path.splitext(file_name)[0]

    if args.log:
        log_name = args.log
    else:
        log_name = f"{file_prefix}-{args.model}-keypoints.json"
    if args.output:
        output_name = args.output
    else:
        output_name = f"{file_prefix}-{args.model}-output.mp4"

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    output_video = cv2.VideoWriter(output_name, fourcc, fps, (source_width, source_height))

    frame_count = 0
    frames_dict = {}
    while True:
        ret_val, image = cap.read()
        if ret_val:
            humans = e.inference(image, resize_to_default=True, upsample_size=4)
            if not args.showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', image)
            fps_time = time.time()
            output_video.write(image)
            if cv2.waitKey(1) == 27:
                break

            frames_dict[str(frame_count)] = {"humans": humans_to_keypoints_dict(humans)}
            frame_count += 1
        else:
            break

    cv2.destroyAllWindows()
    output_video.release()

    scene_dict = {
        "datetime": datetime.utcnow().strftime("%Y%m%d-%H%M"),
        "file_name": file_name,
        "model_height": h,
        "model_width": w,
        "source_height": source_height,
        "source_width": source_width,
        "model": args.model,
        "fps": fps,
        "frames": frames_dict,
    }

    with open(log_name, "w") as f:
        json.dump(scene_dict, f, indent=4)

logger.debug('finished+')
