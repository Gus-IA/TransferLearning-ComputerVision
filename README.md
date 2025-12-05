# Clasificación de Imágenes con Transfer Learning en PyTorch

Este proyecto muestra cómo entrenar un modelo de clasificación de imágenes usando **ResNet18 preentrenado** con PyTorch, aplicando dos estrategias principales:

1. **Fine-tuning completo:** se ajustan todos los pesos del modelo.
2. **Feature extraction:** se congelan las capas convolucionales preentrenadas y solo se entrena el clasificador final.

Se utiliza un dataset de ejemplo llamado `hymenoptera_data` con dos clases (`ants` y `bees`) dividido en carpetas `train` y `val`.

---

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.

