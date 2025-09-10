# modelo_spam.py
# Autor: Leny Lopez - Curso 802 - Materia Machine Learning
# Clasificación de correos SPAM vs HAM con Regresión Logística

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve
)

# =====================
# 1. Cargar dataset
# =====================
df = pd.read_csv("dataset_spam_ham.csv")

print("Primeras filas del dataset:")
print(df.head())

# =====================
# 2. Preprocesamiento
# =====================
X = df.drop("Etiqueta", axis=1)
y = df["Etiqueta"]

# Codificar variable categórica (PaisOrigen)
if "PaisOrigen" in X.columns:
    le = LabelEncoder()
    X["PaisOrigen"] = le.fit_transform(X["PaisOrigen"])

# Escalado de features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================
# 3. División de datos
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# 4. Entrenamiento modelo
# =====================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# =====================
# 5. Evaluación
# =====================
y_probs = model.predict_proba(X_test)[:, 1]

# Umbral óptimo basado en F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

y_pred_opt = (y_probs >= best_threshold).astype(int)

# Métricas
f1 = f1_score(y_test, y_pred_opt)
roc_auc = roc_auc_score(y_test, y_probs)
cm = confusion_matrix(y_test, y_pred_opt)

print("\n=== Resultados del modelo ===")
print(f"Umbral óptimo: {best_threshold:.2f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print("Matriz de confusión:")
print(cm)

# =====================
# 6. Importancia de features
# =====================
features = X.columns
importance = np.abs(model.coef_[0])
importance = importance / importance.sum() * 100  # en %

feat_importance = pd.DataFrame({
    "Feature": features,
    "Importancia (%)": importance
}).sort_values("Importancia (%)", ascending=False)

print("\n=== Importancia de Features (%) ===")
print(feat_importance)

# Gráfico de importancia
plt.figure(figsize=(8,5))
sns.barplot(
    x="Importancia (%)", y="Feature",
    data=feat_importance, palette="viridis"
)
plt.title("Importancia de las características en el modelo")
plt.tight_layout()
plt.savefig("importancia_features.png")
plt.show()

# =====================
# 7. Correlación
# =====================
df_corr = df.copy()
if "PaisOrigen" in df_corr.columns:
    df_corr["PaisOrigen"] = le.transform(df_corr["PaisOrigen"])

corr = df_corr.corr()

# Gráfico Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.tight_layout()
plt.savefig("matriz_correlacion.png")
plt.show()

print("\n=== Correlación con la etiqueta SPAM ===")
print(corr["Etiqueta"].sort_values(ascending=False))
