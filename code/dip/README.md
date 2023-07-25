# Enfoques basados en Deep Image Prior

- **BCDNET_E1**: reconstrucción de la imagen a partir ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en RGB respecto de la imagen original.

- **BCDNET_E2**: reconstrucción de la imagen a partir ruido aleatorio empleando la salida de BCD-NET. Función de pérdida basada en MSE para la reconstrucción en RGB respecto de la imagen original y la diferencia de la matriz de color respecto de la matriz de Ruifrok.

- **CNET_E2**: reconstrucción de la imagen a partir de ruido aleatorio empleando la salida de la C-NET como matrices de concentración y un muestreo aleatorio de la matriz de Ruifrok (std=0.05) como matriz de color. Función de pérdida basada en MSE para la reconstrucción en RGB respecto de la imagen original.
