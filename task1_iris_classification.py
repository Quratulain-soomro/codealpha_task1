# ============================================================
# TASK 1: Iris Flower Classification — CodeAlpha Internship
# ============================================================

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load Dataset ───────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
target_names = iris.target_names

print("=== Dataset Overview ===")
print(f"Shape: {X.shape}")
print(f"\nFeatures:\n{X.describe().round(2)}")
print(f"\nClass distribution:\n{y.value_counts()}")

# ─── 2. Exploratory Data Analysis ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Iris Dataset — Feature Distributions by Species', fontsize=15, fontweight='bold')

colors = ['#4C72B0', '#55A868', '#C44E52']
for ax, feature in zip(axes.flatten(), iris.feature_names):
    for i, (species, color) in enumerate(zip(target_names, colors)):
        ax.hist(X[feature][(y == i).values], bins=15, alpha=0.7, color=color, label=species, edgecolor='white')
    ax.set_title(feature.replace(' (cm)', '').title(), fontsize=11)
    ax.set_xlabel('cm')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('iris_distributions.png', dpi=150, bbox_inches='tight')
plt.close()

# Pairplot-style scatter matrix
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle('Iris — Pairwise Feature Scatter Plots', fontsize=14, fontweight='bold')
feature_pairs = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 3), (2, 3)
]
for ax, (i, j) in zip(axes.flatten(), feature_pairs):
    for k, (species, color) in enumerate(zip(target_names, colors)):
        mask = (y == k).values
        ax.scatter(X.iloc[mask, i], X.iloc[mask, j], c=color, label=species, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(iris.feature_names[i].replace(' (cm)', ''), fontsize=9)
    ax.set_ylabel(iris.feature_names[j].replace(' (cm)', ''), fontsize=9)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('iris_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ─── 3. Preprocessing ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ─── 4. Train & Compare Models ─────────────────────────────
models = {
    'K-Nearest Neighbors':  KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n=== Model Comparison (5-Fold Cross-Validation on Train Set) ===")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    model.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))
    results[name] = {'CV Mean': cv_scores.mean(), 'CV Std': cv_scores.std(), 'Test Acc': test_acc}
    print(f"  {name:30s} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test: {test_acc:.4f}")

# ─── 5. Best Model Detail (SVM) ────────────────────────────
best_model = models['Support Vector Machine']
y_pred = best_model.predict(X_test_sc)

print("\n=== Best Model: Support Vector Machine ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# ─── 6. Visualise Results ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix — SVM', fontsize=13, fontweight='bold')

# Model accuracy comparison
names = list(results.keys())
test_accs = [results[n]['Test Acc'] for n in names]
cv_means  = [results[n]['CV Mean'] for n in names]
x = np.arange(len(names))
w = 0.35
axes[1].bar(x - w/2, cv_means, w, label='CV Mean', color='#4C72B0', alpha=0.85)
axes[1].bar(x + w/2, test_accs, w, label='Test Acc', color='#55A868', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
axes[1].set_ylim(0.85, 1.02)
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Model Comparison', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].axhline(1.0, color='gray', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('iris_results.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance from Random Forest
rf = models['Random Forest']
importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': iris.feature_names, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(feat_df['Feature'], feat_df['Importance'], color=['#4C72B0','#55A868','#C44E52','#8172B2'])
ax.set_xlabel('Importance Score')
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
for bar, val in zip(bars, feat_df['Importance']):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('iris_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n✅ All Task 1 plots saved successfully!")
