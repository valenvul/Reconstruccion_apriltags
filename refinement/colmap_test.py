from colmap_setup import *

captures_dir = "../data/stereo/captures/raiz_apriltags_ordenadas"
db_path = "../data/colmap/database.db"

create_db(db_path)
add_cameras(db_path)

