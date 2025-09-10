# Clasificación de Correos SPAM vs HAM


---

## Contenido del Proyecto

- `dataset_spam_ham.csv` → Dataset con 10 features y la etiqueta (`Etiqueta` = 0 HAM, 1 SPAM).
- `modelo_spam.py` → Script en Python para cargar datos, entrenar el modelo y evaluar métricas.
- `informe_modelo_spam_compacto.pdf` → Informe final con análisis completo.
- `README.md` → Documento actual de referencia.

---

## Dataset

El dataset contiene 10 variables:

1. `LongitudCorreo` → Número total de caracteres.  
2. `NumeroPalabras` → Conteo de palabras.  
3. `NumeroLinks` → Cantidad de enlaces en el correo.  
4. `NumeroAdjuntos` → Archivos adjuntos.  
5. `PorcentajeMayusculas` → Proporción de letras mayúsculas.  
6. `ContieneHTML` → Indicador de si contiene HTML.  
7. `NumeroPalabrasSpam` → Palabras sospechosas (ej. "gratis", "oferta").  
8. `FrecuenciaSignos` → Número de signos de exclamación o similares.  
9. `PaisOrigen` → País desde donde se envió.  
10. `HoraEnvio` → Hora de envío (0–23).  

**Etiqueta (`Etiqueta`)**:  
- `0` = HAM (correo legítimo)  
- `1` = SPAM  

---

## Metodología

1. **Preprocesamiento de datos**
   - Carga del dataset.
   - Codificación de variables categóricas (`PaisOrigen`).
   - Normalización de features.

2. **Entrenamiento**
   - Modelo: `LogisticRegression` de `scikit-learn`.
   - División en entrenamiento (80%) y prueba (20%).

3. **Evaluación**
   - Métricas: F1-Score, ROC-AUC, Matriz de Confusión.
   - Ajuste de **umbral óptimo** para maximizar el F1-Score.

---

## Resultados

- **Umbral óptimo:** ~0.42  
- **F1-Score:** 0.94  
- **ROC-AUC:** 0.97  
- **Matriz de Confusión:**
  - Verdaderos Negativos (TN): 99  
  - Falsos Positivos (FP): 7  
  - Falsos Negativos (FN): 5  
  - Verdaderos Positivos (TP): 89  

---

## Importancia de Features

El modelo asignó los siguientes pesos relativos (porcentaje):

| Feature              | Importancia (%) |
|----------------------|-----------------|
| NumeroPalabrasSpam   | 28.7% |
| NumeroLinks          | 22.1% |
| PorcentajeMayusculas | 19.4% |
| LongitudCorreo       | 9.2% |
| NumeroPalabras       | 7.5% |
| ContieneHTML         | 4.3% |
| NumeroAdjuntos       | 3.8% |
| FrecuenciaSignos     | 2.7% |
| PaisOrigen           | 1.5% |
| HoraEnvio            | 1.0% |

---

## Conclusiones

- El **número de palabras sospechosas**, la **cantidad de enlaces** y el **uso excesivo de mayúsculas** son las variables más influyentes en la clasificación.  
- Todas las 10 características fueron útiles y se mantuvieron en el modelo.  
- El modelo alcanzó un **F1=0.94**, lo cual indica un excelente balance entre precisión y exhaustividad.  
- En escenarios reales, una mejor codificación del `PaisOrigen` podría aumentar aún más el rendimiento.

---

## Cómo Ejecutar

1. Clonar el repositorio o copiar los archivos.  
2. Instalar dependencias:  
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
