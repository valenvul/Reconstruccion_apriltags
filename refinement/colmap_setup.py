import pycolmap
import numpy as np

from refinement.colmap_database import COLMAPDatabase
from utils import read_pickle

def create_db(db_path):
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    db.commit()
    db.close()

def add_cameras(db_path):

    db = pycolmap.Database.open(db_path)

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

    left_camera = pycolmap.camera.create(
        camra_id = 1,
        model = pycolmap.CameraModelId.PINHOLE,
        focal_length_x = fx,
        focal_length_y = fy,
        principal_point_x = cx,
        pricipal_point_y = cy,
        width = image_size[0],
        height = image_size[1],
    )

    left_camera_ok = left_camera.verify_params()

    fx = right_K[0][0]
    fy = right_K[1][1]
    cx = right_K[0][2]
    cy = right_K[1][2]

    right_camera = pycolmap.camera.create(
        camra_id=2,
        model=pycolmap.CameraModelId.PINHOLE,
        focal_length_x=fx,
        focal_length_y=fy,
        principal_point_x=cx,
        pricipal_point_y=cy,
        width=image_size[0],
        height=image_size[1],
    )

    right_camera_ok = right_camera.verify_params()

    db.commit()
    db.close()
    return 1, 2

def add_image(db_path, image, camera_id, qvec, tvec):
    db =  COLMAPDatabase.connect(db_path)

    image_id = db.add_image(name = image, camera_id=camera_id)
    #db.add_keypoints(image_id, keypoints)
    pose = np.hstack([tvec, qvec])
    db.add_pose_prior(image_id, pose)

    db.commit()
    db.close()
    return image_id

def add_keypoints(db_path, image_id, keypoints):
    db = COLMAPDatabase.connect(db_path)

    db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def add_matches(db_path, image_id1, image_id2, matches):
    db = COLMAPDatabase.connect(db_path)

    db.add_matches(image_id1, image_id2, matches)

    db.commit()
    db.close()