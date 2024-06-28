import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# memuat data dari ai4i2020.csv
data = pd.read_csv('ai4i2020.csv')

# menampilkan beberapa baris kolom pada data
print(data.head())
print(data.info())
print(data.describe())

# memeriksa nilai yang kosong/hilang
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# kalau ada data yang hilang maka di hapus
data = data.dropna()

# visualisasi data
plt.figure(figsize=(12, 6))
sns.histplot(data['Tool wear [min]'], kde=True)
plt.title('Tool Wear Distribution')
plt.show()

# visualisasi korelasi
plt.figure(figsize=(12, 6))
# hanya memilih kolom numerik
numerical_data = data.select_dtypes(include=['float', 'int']) 
sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# menentukan x dan y
X = data.drop(columns=['Machine failure', 'UDI', 'Product ID'])
y = data['Machine failure']

# mengidentifikasi kolom non-numeric
non_numeric_cols = X.select_dtypes(exclude=['float', 'int']).columns
print("Non-numeric columns:", non_numeric_cols) # Print these columns to check

# menghapus kolom non-numeric 
X = X.drop(columns=non_numeric_cols) 

# membagi data untuk training dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# normalisasi
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Pemilihan model dan parameter (random forest classifier)
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
# model terbaik
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Evaluasi model
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Print evaluation metrics
print(f'Random Forest Classifier:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(' Random Forest Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# grafik probabilitas kegagalan berdasarkan "Air Temperature [K]" untuk setiap kegagalan
for failure_type in y_test.unique():
    # Filter the data for the current failure type
    df_failure = data[data['Machine failure'] == failure_type]

    # Create a figure
    plt.figure(figsize=(8, 4))

    # grafik probabilitas kegagalan berdasarkan "Air temperature [K]"
    sns.kdeplot(data=df_failure, x='Air temperature [K]')
    plt.title(f'Probabilitas kegagalan berdasarkan Air Temperature [K] ({failure_type})')
    plt.ylabel('Probability Density')

    # Show the plot
    plt.show()