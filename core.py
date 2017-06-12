# import the necessary packages
from __future__ import print_function
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser(description='Human Detector, process video or images.')
ap.add_argument("-i", "--images", required=False, help="path to images directory")
ap.add_argument("-v", "--video", required=False, help="path to video")
ap.add_argument("-b", "--body", action='store_const', const=True, required=False, default=True,
                help="set to true if body detection needed")
args = vars(ap.parse_args())

detect = __import__("detect_human")
if args["images"]:
    if args["body"] is not None:
        detect.__main__(args["images"], True)
    else:
        detect.__main__(args["images"])
else:
    path = os.path.abspath(args.video, args["body"])
    detect.__main__(path)
