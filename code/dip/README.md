# Enfoques basados en Deep Image Prior

- **BCDNET_E1**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en OD respecto de la imagen original en OD. En el documento de la memoria, este modelo recibe el nombre de "Modelo B".

- **BCDNET_E2**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en OD respecto de la imagen original en OD y la diferencia de la matriz de color respecto de la matriz de Ruifrok usando la divergencia de Kullback-Leiber. En el documento de la memoria, este modelo recibe el nombre de "Modelo C sin pre-entrenamiento".

- **BCDNET_E3**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Se tiene un primer periodo durante el entrenamiento en el que la reconstrucción de los colores tiene casi todo el peso de la función de pérdida. Acabado este periodo, se emplea como función de pérdida únicamente el MSE para la reconstrucción en OD respecto de la imagen original en OD. En el documento de la memoria, este modelo recibe el nombre de "Modelo C con pre-entrenamiento".

- **CNET_E2**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de la C-NET como matrices de concentración y un muestreo aleatorio de la matriz de Ruifrok (std=0.05) como matriz de color. Función de pérdida basada en MSE para la reconstrucción en RGB respecto de la imagen original. En el documento de la memoria, este modelo recibe el nombre de "Modelo A".
