import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Charger le jeu de données
data = pd.read_csv('farms_train.csv', delimiter=';', decimal=',')

# Supprimer toutes les lignes contenant des valeurs NaN
data = data.dropna()
    
# Séparer les caractéristiques (X) et la variable cible (y)
X = data.drop(columns=['DIFF'])
y = data['DIFF']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraîner le modèle SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Faire des prédictions sur l'ensemble de test
y_pred = svm_classifier.predict(X_test_scaled)

# Imprimer le rapport de classification
report = classification_report(y_test, y_pred)
print(report)