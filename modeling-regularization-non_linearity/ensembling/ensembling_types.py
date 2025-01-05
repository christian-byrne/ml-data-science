from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Train individual models
rf = RandomForestClassifier(n_estimators=50, random_state=42)
gb = GradientBoostingClassifier(n_estimators=50, random_state=42)

rf.fit(X, y)
gb.fit(X, y)

# Combine predictions (simple voting ensemble)
rf_preds = rf.predict(X)
gb_preds = gb.predict(X)
ensemble_preds = (rf_preds + gb_preds) > 1  # Majority vote

# Evaluate accuracy
accuracy = accuracy_score(y, ensemble_preds)
print(f"Ensemble Accuracy: {accuracy:.4f}")
