# import the necessary packages
from __future__ import print_function
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=False, help="path to images directory")
ap.add_argument("-v", "--video", required=False, help="path to video")
args = vars(ap.parse_args())

detect = __import__("detect_human")
if args["images"]:
    detect.__main__(args["images"])
else:
    detect.__main__(args["video"])
