# Enfoques basados en Deep Image Prior

- **BCDNET_E1**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en OD respecto de la imagen original en OD. 

- **BCDNET_E2**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en OD respecto de la imagen original en OD y la diferencia de la matriz de color respecto de la matriz de Ruifrok usando la divergencia de Kullback-Leiber.

- **BCDNET_E3**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Se tiene un primer periodo durante el entrenamiento en el que la reconstrucción de los colores tiene casi todo el peso de la función de pérdida. Acabado este periodo, se emplea como función de pérdida únicamente el MSE para la reconstrucción en OD respecto de la imagen original en OD. No he probado a emplear la ponderación habitual, ya que en el caso de BCDNET_E2 no parecía influir.

- **BCDNET_E4**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de BCD-NET. Este enfoque incorpora el uso de la norma L2 para intentar obtener resultados más suavizados, en tanto que tendrá menor valor si se usa o bien H o bien E y no ambas. Dado que los errores eran enormes dividí dicho valor entre el número de píxeles y le incorporé el MSE de reconstrucción y Kullback-Leiber a misma ponderación todo.

- **CNET_E2**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de la C-NET como matrices de concentración y un muestreo aleatorio de la matriz de Ruifrok (std=0.05) como matriz de color. Función de pérdida basada en MSE para la reconstrucción en RGB respecto de la imagen original.
