import pycolmap
from pathlib import Path

from refinement.colmap_setup import create_db

output_path = Path("prueba")
image_path = "data/stereo/captures/raiz_apriltags_ordenadas"
database_path = output_path / "database.db"
sfm_path = output_path / "sfm"

output_path.mkdir(exist_ok=True)

create_db(database_path)
pycolmap.import_images(database_path, image_path)
pycolmap.extract_features(database_path, image_path)
pycolmap.match_exhaustive(database_path)

