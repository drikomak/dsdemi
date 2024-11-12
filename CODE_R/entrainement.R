
# Installer le package e1071 POU
# install.packages("caret")
# Charger le package

library(e1071)
source("library.R")

library(pROC)
library(caret)
library(ggplot2)
# Charger le jeu de données
data1 <- read.csv2("farms_train.csv") %>%
  filter(!is.na(DIFF))  # Supprimer les lignes où DIFF est NA

# S'assurer que DIFF est binaire (0 ou 1) et que c'est un facteur
data1$DIFF <- as.factor(data1$DIFF)

# Vous pouvez aussi vérifier si les autres variables explicatives sont numériques
data1$R2 <- as.numeric(data1$R2)
data1$R7 <- as.numeric(data1$R7)
data1$R8 <- as.numeric(data1$R8)
data1$R17 <- as.numeric(data1$R17)
data1$R22 <- as.numeric(data1$R22)
data1$R32 <- as.numeric(data1$R32)


# Diviser les données en ensembles d'entraînement et de test
set.seed(42)  # Fixer la graine pour la reproductibilité
train_indices <- sample(1:nrow(data1), size = 0.8 * nrow(data1))  # 80% pour l'entraînement
train_data <- data1[train_indices, ]
test_data <- data1[-train_indices, ]


# # Normaliser les variables explicatives (R2, R7, R8, R17, R22, R32)
# scaler <- scale(train_data[, c("R2", "R7", "R8", "R17", "R22", "R32")])
# train_data[, c("R2", "R7", "R8", "R17", "R22", "R32")] <- scaler
# 
# # Appliquer la même normalisation aux données de test
# scaler_test <- scale(test_data[, c("R2", "R7", "R8", "R17", "R22", "R32")], center = attr(scaler, "scaled:center"), scale = attr(scaler, "scaled:scale"))
# test_data[, c("R2", "R7", "R8", "R17", "R22", "R32")] <- scaler_test



# Entraîner un modèle SVM
svm_model <- svm(DIFF ~ R2 + R7 + R8 + R17 + R22 + R32, data = train_data, kernel = "linear", type = "C-classification")

# Résumé du modèle
summary(svm_model)



# Prédire sur l'ensemble de test
predictions <- predict(svm_model, newdata = test_data)

# Calculer la précision
conf_matrix <- table(Predicted = predictions, Actual = test_data$DIFF)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Précision du modèle : ", accuracy * 100, "%"))


# Calculer la matrice de confusion
conf_matrix <- confusionMatrix(predictions, test_data$DIFF)

# Extraire la table de la matrice de confusion
conf_matrix_plot <- as.data.frame(conf_matrix$table)

# Extraire les métriques de performance à partir de la matrice de confusion
# Extraire les métriques de la matrice de confusion
precision <- conf_matrix$byClass["Pos Pred Value"]
recall <- conf_matrix$byClass["Sensitivity"]
f_measure <- conf_matrix$byClass["F1"]

# Afficher les résultats
print(paste("Précision : ", round(precision, 2)))
print(paste("Rappel : ", round(recall, 2)))
print(paste("F-mesure : ", round(f_measure, 2)))








# Effectuer l'ACP sur les variables explicatives (R2, R7, R8, R17, R22, R32)
acp_result <- prcomp(train_data[, c("R2", "R7", "R8", "R17", "R22", "R32")], scale. = TRUE)

# Résumé de l'ACP
summary(acp_result)

# Visualisation des variances expliquées par chaque composante
variances <- acp_result$sdev^2 / sum(acp_result$sdev^2)  # Variance expliquée par chaque composante
cumulative_variances <- cumsum(variances)  # Variance cumulative
# Tracer la variance expliquée
plot(variances, type = "b", main = "Variance expliquée par chaque composante", 
     xlab = "Composantes Principales", ylab = "Variance expliquée", pch = 19)
abline(h = 0.9, col = "red", lty = 2)  # Ligne pour montrer la variance cumulative de 90%
# Obtenir la projection des variables sur les composantes principales
variables_acp <- acp_result$rotation  # Coefficients des variables pour chaque composante



variables_acp_df <- as.data.frame(variables_acp)
variables_acp_df$Variable <- rownames(variables_acp_df)

# Plot avec ggplot
ggplot(variables_acp_df, aes(x = PC1, y = PC2, label = Variable)) +
  geom_text() +
  labs(title = "Projection des variables sur les 2 premières composantes principales",
       x = "Composante Principale 1", y = "Composante Principale 2") +
  theme_minimal()









# Projection des individus (observations) sur les composantes principales (à partir de train_data)
individus_acp <- acp_result$x  # Projection des individus sur les composantes principales

# Créer un dataframe pour ggplot
individus_acp_df <- as.data.frame(individus_acp)

# Ajouter la variable DIFF de train_data
individus_acp_df$DIFF <- train_data$DIFF

# Visualiser les individus sur les 2 premières composantes principales
ggplot(individus_acp_df, aes(x = PC1, y = PC2, color = DIFF)) +
  geom_point() +  # Ajouter les points
  labs(title = "Projection des individus de l'ensemble d'entraînement sur les 2 premières composantes principales",
       x = "Composante Principale 1", y = "Composante Principale 2") +
  theme_minimal() +
  scale_color_manual(values = c("red", "blue"))  # Vous pouvez changer les couleurs








data2 <- read.csv2("farms_test.csv") %>%
  filter(!is.na(R2))

# Assurez-vous que les variables explicatives de data2 sont numériques
data2$R2 <- as.numeric(data2$R2)
data2$R7 <- as.numeric(data2$R7)
data2$R8 <- as.numeric(data2$R8)
data2$R17 <- as.numeric(data2$R17)
data2$R22 <- as.numeric(data2$R22)
data2$R32 <- as.numeric(data2$R32)

# # Normaliser data2 avec les mêmes paramètres que ceux utilisés pour train_data
# scaler_data2 <- scale(data2[, c("R2", "R7", "R8", "R17", "R22", "R32")],
#                       center = attr(scaler, "scaled:center"),
#                       scale = attr(scaler, "scaled:scale"))
# 
# # Appliquer la normalisation à data2
# data2[, c("R2", "R7", "R8", "R17", "R22", "R32")] <- scaler_data2

# Prédire la variable DIFF avec le modèle SVM entraîné
predictions_data2 <- predict(svm_model, newdata = data2)

# Ajouter les prédictions à data2
data2$Predicted_DIFF <- predictions_data2


# Prédictions de probabilité (probabilité de chaque classe)
predictions_prob <- predict(svm_model, newdata = test_data, decision.values = TRUE)

# Extraire les valeurs de probabilité pour la classe positive (1)
prob_positive_class <- attributes(predictions_prob)$decision.values

# Calculer la courbe ROC
roc_curve <- roc(test_data$DIFF, prob_positive_class)

# Visualiser la courbe ROC
plot(roc_curve, main = "Courbe ROC - Performance du modèle SVM",
     col = "blue", lwd = 2)





# Prédictions du modèle (classe prédite 0 ou 1)
predictions <- predict(svm_model, newdata = test_data)

# Créer un dataframe avec les prédictions et les valeurs réelles
results_df <- data.frame(
  Predicted = predictions,
  Actual = test_data$DIFF
)

# Visualiser avec ggplot
ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_jitter(width = 0.1, height = 0.1, color = "blue") +
  labs(title = "Prédictions vs. Réels - Modèle SVM",
       x = "Valeurs Réelles (DIFF)",
       y = "Prédictions (DIFF)") +
  theme_minimal() +
  theme(legend.position = "none") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed")










# Créer la matrice de confusion
conf_matrix <- confusionMatrix(predictions, data2$DIFF)

# Afficher la matrice de confusion
print(conf_matrix)

# Extraire les principales métriques
accuracy <- conf_matrix$overall["Accuracy"]
recall <- conf_matrix$byClass["Recall"]
precision <- conf_matrix$byClass["Precision"]
f1_score <- conf_matrix$byClass["F1"]

# Afficher les résultats
cat("Accuracy: ", accuracy, "\n")
cat("Recall: ", recall, "\n")
cat("Precision: ", precision, "\n")
cat("F1 Score: ", f1_score, "\n")

