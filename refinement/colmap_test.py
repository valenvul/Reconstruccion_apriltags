import cv2
import pycolmap
import open3d as o3d
import shutil
from pathlib import Path


from refinement.colmap_setup import *
from images import prepare_imgs
from calibration.calib import *

output_path = Path("../colmap/")
image_path = "../data/stereo/captures/raiz_apriltags_ordenadas"
db_path = output_path / "database.db"
sfm_path = output_path / "sfm"

captures_dir = "../data/stereo/captures/raiz_apriltags_ordenadas"
tag_family = "tag25h9"

output_path.mkdir(exist_ok=True)


if not os.path.exists(
            db_path
    ):
    create_db(db_path)
    id_left_cam, id_right_cam = add_cameras(db_path)

    #### add images

    left_file_names, right_file_names = prepare_imgs(captures_dir)

    calib_file = "../data/stereo/stereo_calibration.pkl"
    maps_file = "../data/stereo/stereo_maps.pkl"
    calibration = read_pickle(calib_file)
    maps = read_pickle(maps_file)

    # separate calibration params
    left_K = calibration["left_K"]
    left_dist = calibration["left_dist"]
    right_K = calibration["right_K"]
    right_dist = calibration["right_dist"]
    image_size = calibration["image_size"]
    T = calibration["T"]

    left_map_x = maps["left_map_x"]
    left_map_y = maps["left_map_y"]
    right_map_x = maps["right_map_x"]
    right_map_y = maps["right_map_y"]
    P1 = maps["P1"]
    P2 = maps["P2"]
    Q = maps["Q"]

    apriltags_correspondences = np.full((30, 9), -1, dtype=object)

    image_num = -1
    for left_file_name, right_file_name in zip(
            left_file_names, right_file_names):
            image_num += 1

            image_size, left_image, right_image = process_images(left_file_name, right_file_name, image_size)
            left_size = (left_image.shape[1], left_image.shape[0])
            right_size = (right_image.shape[1], right_image.shape[0])

            # rectify images
            left_image_rectified = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
            right_image_rectified = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)

            ######### POSE ##################

            left_object_points, left_image_points, left_tag_dict = detect_apriltags(left_image_rectified,tag_family)
            right_object_points, right_image_points, right_tag_dict = detect_apriltags(right_image_rectified, tag_family)

            left_color = cv2.cvtColor(left_image_rectified, cv2.COLOR_GRAY2BGR)
            right_color = cv2.cvtColor(right_image_rectified, cv2.COLOR_GRAY2BGR)

            # Verificar si se encontraron suficientes puntos
            if len(left_image_points) < 1 or len(right_image_points) < 1:
                print("No se encontraron suficientes AprilTags en ambas imágenes.")
                continue

            # rvec rota puntos del sistema de coordenadas del objeto al sistema de coordenadas de la cámara
            # rvec is the rotation vector and tvec is the movement vector of the cameras in respect to the world origin
            ret, rvec, tvec = cv2.solvePnP(
                left_object_points,
                left_image_points,
                left_K,
                left_dist,
                flags=cv2.SOLVEPNP_EPNP
            )

            left_image_id = add_image(db_path, left_image_rectified, id_left_cam, rvec, tvec)
            right_image_id = add_image(db_path, right_image_rectified, id_right_cam, rvec, tvec)

            # add_keypoints(db_path, left_image_id, left_image_points)
            # add_keypoints(db_path, right_image_id, right_image_points)

            # add all detections to the matrix to have the correspondences
            for i in range(8):
                if i in left_tag_dict:
                    apriltags_correspondences[image_num][i] = left_tag_dict[i]
                    apriltags_correspondences[image_num][8] = left_image_id
                if i in right_tag_dict:
                    apriltags_correspondences[image_num+15][i] = right_tag_dict[i]
                    apriltags_correspondences[image_num+15][8] = right_image_id


    # add keypoints and matches for all images from correspondance matrix

    # para cada imagen i
        # para cada otra imagen j
            # Para cada apriltag
                # si hay un elemento en i lo agrego a keypoints de i y aumento el indice de i
                # si hay un elemento en j aumento el indice de j
                # si hay un elemento en los dos pongo la tupla i,j en matches
            #agrego los keypoints
            # agrego los matches

    for i in range(30):
        id_i = apriltags_correspondences[i][8]
        index_i = -1
        keypoints = []
        for k in range(8):
            if not isinstance(apriltags_correspondences[i][k], int):
                keypoints.append(apriltags_correspondences[i][k])
        keypoints = np.array(keypoints)
        add_keypoints(db_path, id_i, keypoints)
        for j in range(i, 30):
            if j!= i:
                id_j = apriltags_correspondences[j][8]
                matches = []
                index_j = -1
                for apriltag in range(8):
                    if not isinstance(apriltags_correspondences[i][apriltag],int):
                        index_i += 1
                    if not isinstance(apriltags_correspondences[j][apriltag],int):
                        index_j += 1
                    if not isinstance(apriltags_correspondences[i][apriltag],int) and not isinstance(apriltags_correspondences[j][apriltag],int):
                        index_tuple = [index_i, index_j]
                        matches.append(index_tuple)
                if len(matches) > 0:
                    matches = np.array(matches)
                    add_matches(db_path, id_i, id_j, matches)

####### Reconstrucción
if sfm_path.exists():
    shutil.rmtree(sfm_path)
sfm_path.mkdir(exist_ok=True)

# # Ejecutar la reconstrucción incremental
recs = pycolmap.incremental_mapping(
    db_path,
    image_path,
    sfm_path
)

# # Ahora puedes cargar la reconstrucción correctamente
reconstruction = pycolmap.Reconstruction(sfm_path)

# bundle adjustment
ba_options = pycolmap.BundleAdjustmentOptions()
ba_options.refine_focal_length = True  # Mantener fijos los intrínsecos
ba_options.refine_principal_point = False
ba_options.refine_extra_params = False
ba_options.refine_extrinsics = True

pycolmap.bundle_adjustment(reconstruction, ba_options)

reconstruction.write(output_path)

ply_path = "colmap/sparse.ply"
reconstruction.export_PLY(ply_path)

# Cargar la nube de puntos desde el archivo .ply
point_cloud = o3d.io.read_point_cloud(ply_path)

# Visualizar la nube de puntos
o3d.visualization.draw_geometries([point_cloud])



