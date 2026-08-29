from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. Load Iris Dataset
# ==========================================

iris = load_iris()

X = iris.data
y = iris.target

print("==========================================")
print("AI-Based Iris Flower Classification")
print("==========================================")

print("\nDataset loaded successfully!")
print("Dataset shape:", X.shape)
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)


# ==========================================
# 2. Train-Test Split
# 80% Training / 20% Testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ==========================================
# 3. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")


# ==========================================
# 4. Find Best K
# ==========================================

error_rates = []

for k in range(1, 16):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)

    error_rate = (predictions != y_test).mean()

    error_rates.append(error_rate)


# ==========================================
# 5. Display Error Rates
# ==========================================

print("\nK Values and Error Rates:")

for k, error in zip(range(1, 16), error_rates):
    print(f"K = {k}, Error Rate = {error:.4f}")


# ==========================================
# 6. Find Best K
# ==========================================

best_k = range(1, 16)[error_rates.index(min(error_rates))]

print("\nBest K:", best_k)
print("Lowest Error Rate:", f"{min(error_rates):.4f}")


# ==========================================
# 7. Save KNN Error Rate Graph
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 16),
    error_rates,
    marker="o"
)

plt.title("KNN Error Rate for Different K Values")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.xticks(range(1, 16))
plt.grid(True)

plt.tight_layout()

# Save graph instead of opening it every time
plt.savefig("knn_error_rate.png", dpi=300)

plt.close()

print("\nKNN error-rate graph saved as: knn_error_rate.png")


# ==========================================
# 8. Train Final KNN Model
# ==========================================

final_knn = KNeighborsClassifier(n_neighbors=best_k)

final_knn.fit(X_train_scaled, y_train)

print("\nFinal KNN model trained successfully!")


# ==========================================
# 9. Make Predictions
# ==========================================

y_pred = final_knn.predict(X_test_scaled)

print("Predictions completed!")


# ==========================================
# 10. Calculate Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 12. Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ==========================================
# 13. Save Confusion Matrix
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title("KNN Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.close()

print("\nConfusion matrix saved as: confusion_matrix.png")

print("\n==========================================")
print("PROJECT COMPLETED SUCCESSFULLY!")
print("==========================================")