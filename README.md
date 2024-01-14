# Trabajo Fin de Máster: *Deconvolución Ciega de Imágenes Histológicas Usando Aprendizaje Profundo.*
---
Autor: *José Alberto Gómez García*

Director: *Rafael Molina Soriano*

Codirector: *Fernando Pérez Bueno*

---

En este trabajo se propone la aplicación del enfoque Deep Image Prior a modelos de aprendizaje profundo basados en redes neuronales convolucionales con el objetivo de realizar BCD sobre imágenes histológicas. De esta manera, podemos obtener los beneficios de la aplicación de redes neuronales sin requerir de grandes conjuntos de datos que contengan "ground truth". Para realizar la deconvolución ciega de color de una imagen histológica requeriremos única y exclusivamente de dicha imagen.

Se propondrán tres modelos diferentes, que emplearán arquitecturas, funciones de pérdida y tipos de entrada diferentes. Estos serán puestos a prueba sobre el conjunto de datos ``Warwick Stain Separation Benchmark''. Los experimentos realizados indican que se pueden obtener resultados comparables a los proporcionados por modelos amortizados entrenados en grandes conjuntos de datos. Se propone también una variante de Deep Image Prior en la que las redes neuronales se inicializan con los pesos de un entrenamiento previo; lo que permite mejorar los resultados significativamente. 
