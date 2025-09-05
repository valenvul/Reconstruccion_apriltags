# Recontrucción 3D de raíces

Este es el primer approach a la reconstrucción 3D de raíces crecidas en hidroponia. El proyecto fue realizado junto Apolo biotech y Gastón Castro.

En este primer acercamiento se busca lograr una recontrucción 3D (pointcloud) de un objeto a pertir de imágenes sacadas con una cámara stereo. Primero se calibra la cámara a aprtor de imágenes de un patron de chekerboard. Luego, se usan apriltags posicionados sobre el objeto para poder triangular la posición del mismo en base a las imágenes. Por último, se hace un refinamiento de la nube final con algortimos como COLMAP e ICP.
