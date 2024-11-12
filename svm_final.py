import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


# Charger les données d'entraînement
data = pd.read_csv('farms_train.csv', delimiter=';', decimal=',')

# Nettoyer les données en supprimant les valeurs manquantes
data = data.dropna()

# Séparer les features et la cible
X = data.drop(columns=["DIFF"])
y = data["DIFF"]

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialiser le modèle SVM
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)

# Effectuer la validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = cross_val_score(svm_classifier, X_scaled, y, cv=cv, scoring='accuracy')
precision_scores = cross_val_score(svm_classifier, X_scaled, y, cv=cv, scoring='precision')
recall_scores = cross_val_score(svm_classifier, X_scaled, y, cv=cv, scoring='recall')
f1_scores = cross_val_score(svm_classifier, X_scaled, y, cv=cv, scoring='f1')

# Afficher les métriques de validation croisée
plt.figure(figsize=(14, 7))
plt.plot(range(1, 6), accuracy_scores, marker='o', label='Accuracy', linestyle='-')
plt.plot(range(1, 6), precision_scores, marker='s', label='Precision', linestyle='--')
plt.plot(range(1, 6), recall_scores, marker='^', label='Recall', linestyle='-.')
plt.plot(range(1, 6), f1_scores, marker='d', label='F1 Score', linestyle=':')

plt.title('Cross-Validation Metrics Evolution pour SVM')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.xticks(range(1, 6))
plt.legend()
plt.grid(True)
plt.show()

# Afficher les moyennes et les écarts-types des scores
print(f'Mean Accuracy: {np.mean(accuracy_scores):.2f} ± {np.std(accuracy_scores):.2f}')
print(f'Mean Precision: {np.mean(precision_scores):.2f} ± {np.std(precision_scores):.2f}')
print(f'Mean Recall: {np.mean(recall_scores):.2f} ± {np.std(recall_scores):.2f}')
print(f'Mean F1 Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}')

# Calculer la courbe ROC
y_pred_proba = cross_val_predict(svm_classifier, X_scaled, y, cv=cv, method='predict_proba')
fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
plt.title('Mean ROC Curve for SVM (Cross-Validation)')
plt.show()

# Entraîner le modèle sur l'ensemble des données d'entraînement
svm_classifier.fit(X_scaled, y)

# Charger les nouvelles données de test
test_data = pd.read_csv('farms_test.csv', delimiter=';', decimal=',')

# Nettoyer les données de test
test_data = test_data.dropna()

# Appliquer le même scaler sur les données de test
X_test_scaled = scaler.transform(test_data)

# Faire des prédictions sur les données de test
y_pred = svm_classifier.predict(X_test_scaled)

# Ajouter la colonne 'DIFF' avec les prédictions dans le DataFrame de test
test_data['DIFF'] = y_pred

# Afficher le DataFrame de test avec les prédictions ajoutées
print(test_data)

# Enregistrer le DataFrame de test modifié dans un fichier CSV si nécessaire
test_data.to_csv('farms_test_with_predictions.csv', index=False, sep=';', decimal=',')


print(tabulate(test_data, headers='keys', tablefmt='grid'))


diff_only = test_data[['DIFF']]

# Enregistrer dans un fichier CSV avec une seule colonne "DIFF"
diff_only.to_csv('diff_predictions_only.csv', index=False, sep=';', decimal=',')