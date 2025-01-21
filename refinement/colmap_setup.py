import pycolmap
from images import *
from colmap_database import COLMAPDatabase
from utils import read_pickle

def create_db(path):
    db = COLMAPDatabase.connect(path)
    db.create_tables()

def add_cameras(path):

    db = COLMAPDatabase.connect(path)

    # read calibration files
    calib_file = "../data/stereo/stereo_calibration.pkl"
    calibration = read_pickle(calib_file)

    # separate calibration params
    left_K = calibration["left_K"]
    right_K = calibration["right_K"]
    image_size = calibration["image_size"]

    fx = left_K[0][0]
    fy = left_K[1][1]
    cx = left_K[0][2]
    cy = left_K[1][2]

    db = COLMAPDatabase.connect(path)
    id_left_cam = db.add_camera(
        model = "PINHOLE",
        width = image_size[0],
        height = image_size[1],
        params = [fx, fy, cx, cy]
    )

    print(f"Cámara izquierda añadida con ID: {id_left_cam}")

    fx = right_K[0][0]
    fy = right_K[1][1]
    cx = right_K[0][2]
    cy = right_K[1][2]
    id_right_cam = db.add_camera(
        model = "PINHOLE",
        width = image_size[0],
        height = image_size[1],
        params = [fx, fy, cx, cy]
    )

    print(f"Cámara derecha añadida con ID: {id_right_cam}")

    db.close()


