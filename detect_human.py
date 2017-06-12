# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
import imghdr


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def __main__(path, body=False):

    t_bool = os.path.isdir(path)
    # print(t_bool)
    if t_bool:
        # initialize the HOG descriptor/person detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # loop over the image paths
        for imagePath in paths.list_images(path):
            # load the image and resize it to (1) reduce detection time
            # and (2) improve detection accuracy
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image is not None:

                if body is not False:
                    # detect people in the image
                    (rects, weights) = hog.detectMultiScale(image, winStride=(2, 2), padding=(1, 1), scale=1.05)

                    # draw the original bounding boxes
                    draw_detections(image, rects)

                face_cascade = cv2.CascadeClassifier('detectors/haarcascade_frontalface_default.xml')
                face_extended_cascade = cv2.CascadeClassifier('detectors/haarcascade_profileface.xml')
                face_cascade_alt = cv2.CascadeClassifier('detectors/haarcascade_frontalface_alt.xml')
                face_cascade_alt2 = cv2.CascadeClassifier('detectors/haarcascade_frontalface_alt2.xml')

                faces = face_cascade.detectMultiScale(gray, 1.1, 1)
                for (x, y, w, h) in faces:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]

                faces_extended = face_extended_cascade.detectMultiScale(gray, 1.1, 1)
                for (x, y, w, h) in faces_extended:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]

                faces_alt = face_cascade_alt.detectMultiScale(gray, 1.1, 1)
                for (x, y, w, h) in faces_alt:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]

                faces_alt2 = face_cascade_alt2.detectMultiScale(gray, 1.1, 1)
                for (x, y, w, h) in faces_alt2:
                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = image[y:y+h, x:x+w]

                cv2.imshow("Detection", image)
                cv2.waitKey(0)

    else:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cap = cv2.VideoCapture(path)
        while True:
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
            draw_detections(frame, found)
            cv2.imshow('feed', frame)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        cv2.destroyAllWindows()
