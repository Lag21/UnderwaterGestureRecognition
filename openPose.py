from openpose import pyopenpose as op
from PIL import Image, ImageDraw
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["body"] = 1
params["face"] = False
params["face_detector"] = 1
params["hand"] = False
params["hand_detector"] = 1
params["disable_blending"] = True
params["model_folder"] = "/home/alejandro/openpose_folder/openpose/models/"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()

datum.cvInputData = img