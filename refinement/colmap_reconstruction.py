import shutil

from pathlib import Path

import open3d as o3d

import pycolmap


def run():
    output_path = Path("../example/")
    image_path = "../data/stereo/captures/raiz_apriltags_ordenadas"
    database_path = output_path / "database.db"
    sfm_path = output_path / "sfm"

    output_path.mkdir(exist_ok=True)

    pycolmap.extract_features(database_path, image_path)
    pycolmap.match_exhaustive(database_path)
    num_images = pycolmap.Database(database_path).num_images

    if sfm_path.exists():
        shutil.rmtree(sfm_path)
    sfm_path.mkdir(exist_ok=True)


    recs = pycolmap.incremental_mapping(
        database_path,
        image_path,
        sfm_path
    )

    reconstruction = pycolmap.Reconstruction(sfm_path / "0")


    # Configurar opciones para el Bundle Adjustment
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = True  # Mantener fijos los intrínsecos
    ba_options.refine_principal_point = False
    ba_options.refine_extra_params = False
    ba_options.refine_extrinsics = True

    # Aplicar el Bundle Adjustment
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    # Guardar la reconstrucción optimizada
    reconstruction.write(output_path)

    reconstruction.export_PLY("example/reconstruction.ply")
    # Cargar la nube de puntos desde el archivo .ply
    point_cloud = o3d.io.read_point_cloud("example/reconstruction.ply")

    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([point_cloud])



if __name__ == "__main__":
    run()