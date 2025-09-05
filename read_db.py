
import sqlite3
import numpy as np

def check_table_counts(db_path):
    # Conectar a la base de datos
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Obtener todas las tablas de la base de datos
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Contar los registros en cada tabla
    table_counts = {}
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table};")
        count = cursor.fetchone()[0]
        table_counts[table] = count

    # Cerrar la conexión
    conn.close()

    return table_counts

# Uso de la función
db_path = "colmap/database.db"  # Cambia esto a la ruta de tu archivo
table_counts = check_table_counts(db_path)

# Mostrar resultados
for table, count in table_counts.items():
    print(f"Tabla '{table}': {count} registros")


# Conectar a la base de datos
conn = sqlite3.connect("colmap/database.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM cameras")
cameras = cursor.fetchall()

for cam in cameras:
    print(cam)  # Ver qué modelo de cámara está guardado


conn.close()