import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from backprop_model import BackpropNN
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==================== LOAD & PREPROCESS ====================
df = pd.read_csv('data/gaming_industry_trends.csv')
print(f"Dataset loaded: {df.shape}")

# Encode categorical
le_genre = LabelEncoder()
le_platform = LabelEncoder()
le_dev = LabelEncoder()
le_esports = LabelEncoder()
le_trend = LabelEncoder()

df['Genre_enc'] = le_genre.fit_transform(df['Genre'])
df['Platform_enc'] = le_platform.fit_transform(df['Platform'])
df['Developer_enc'] = le_dev.fit_transform(df['Developer'])
df['Esports_enc'] = le_esports.fit_transform(df['Esports Popularity'])
df['Trend_enc'] = le_trend.fit_transform(df['Trending Status'])

# Features & Target (Prediksi Revenue)
feature_cols = ['Release Year', 'Players (Millions)', 'Peak Concurrent Players',
                'Metacritic Score', 'Genre_enc', 'Platform_enc', 'Developer_enc',
                'Esports_enc', 'Trend_enc']
target_col = 'Revenue (Millions $)'

X = df[feature_cols].values
y = df[target_col].values

# Scale
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

results = {}

# ==================== 1. LINEAR REGRESSION ====================
print("\n[1] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

y_pred_lr_orig = scaler_y.inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae_lr = mean_absolute_error(y_test_orig, y_pred_lr_orig)
rmse_lr = np.sqrt(mean_squared_error(y_test_orig, y_pred_lr_orig))
r2_lr = r2_score(y_test_orig, y_pred_lr_orig)

results['Linear Regression'] = {'MAE': round(mae_lr, 2), 'RMSE': round(rmse_lr, 2), 'R2': round(r2_lr, 4)}
print(f"  MAE: {mae_lr:.2f} | RMSE: {rmse_lr:.2f} | R2: {r2_lr:.4f}")
joblib.dump(lr, 'models/linear_regression.pkl')

# ==================== 2. ANN ====================
print("\n[2] Training ANN...")
ann = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
ann.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_ann = ann.fit(X_train, y_train, epochs=80, batch_size=32,
                      validation_split=0.15, verbose=0)

y_pred_ann = ann.predict(X_test, verbose=0).flatten()
y_pred_ann_orig = scaler_y.inverse_transform(y_pred_ann.reshape(-1, 1)).flatten()

mae_ann = mean_absolute_error(y_test_orig, y_pred_ann_orig)
rmse_ann = np.sqrt(mean_squared_error(y_test_orig, y_pred_ann_orig))
r2_ann = r2_score(y_test_orig, y_pred_ann_orig)

results['ANN'] = {'MAE': round(mae_ann, 2), 'RMSE': round(rmse_ann, 2), 'R2': round(r2_ann, 4)}
print(f"  MAE: {mae_ann:.2f} | RMSE: {rmse_ann:.2f} | R2: {r2_ann:.4f}")
ann.save('models/ann_model.h5')

# Save ANN loss history
ann_loss = {
    'train_loss': [float(v) for v in history_ann.history['loss']],
    'val_loss': [float(v) for v in history_ann.history['val_loss']]
}
with open('models/ann_loss.json', 'w') as f:
    json.dump(ann_loss, f)

# ==================== 3. RNN/LSTM ====================
print("\n[3] Training RNN/LSTM...")
# Reshape for LSTM: (samples, timesteps, features)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

rnn = keras.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[1])),
    layers.LSTM(32),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
rnn.compile(optimizer='adam', loss='mse', metrics=['mae'])
history_rnn = rnn.fit(X_train_rnn, y_train, epochs=80, batch_size=32,
                      validation_split=0.15, verbose=0)

y_pred_rnn = rnn.predict(X_test_rnn, verbose=0).flatten()
y_pred_rnn_orig = scaler_y.inverse_transform(y_pred_rnn.reshape(-1, 1)).flatten()

mae_rnn = mean_absolute_error(y_test_orig, y_pred_rnn_orig)
rmse_rnn = np.sqrt(mean_squared_error(y_test_orig, y_pred_rnn_orig))
r2_rnn = r2_score(y_test_orig, y_pred_rnn_orig)

results['RNN/LSTM'] = {'MAE': round(mae_rnn, 2), 'RMSE': round(rmse_rnn, 2), 'R2': round(r2_rnn, 4)}
print(f"  MAE: {mae_rnn:.2f} | RMSE: {rmse_rnn:.2f} | R2: {r2_rnn:.4f}")
rnn.save('models/rnn_model.h5')

rnn_loss = {
    'train_loss': [float(v) for v in history_rnn.history['loss']],
    'val_loss': [float(v) for v in history_rnn.history['val_loss']]
}
with open('models/rnn_loss.json', 'w') as f:
    json.dump(rnn_loss, f)

# ==================== 4. K-MEANS CLUSTERING ====================
print("\n[4] Training K-Means Clustering...")
kmeans_features = ['Revenue (Millions $)', 'Players (Millions)',
                   'Peak Concurrent Players', 'Metacritic Score']
X_km = df[kmeans_features].values
scaler_km = StandardScaler()
X_km_scaled = scaler_km.fit_transform(X_km)

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_km_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_km_scaled, km.labels_))

# Best K = 4
best_k = 4
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_km_scaled)
sil = silhouette_score(X_km_scaled, kmeans.labels_)

results['K-Means'] = {
    'Inertia': round(kmeans.inertia_, 2),
    'Silhouette': round(sil, 4),
    'K': best_k
}
print(f"  Inertia: {kmeans.inertia_:.2f} | Silhouette: {sil:.4f} | K: {best_k}")
joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(scaler_km, 'models/scaler_km.pkl')

elbow_data = {
    'k': list(K_range),
    'inertia': [round(v, 2) for v in inertias],
    'silhouette': [round(v, 4) for v in sil_scores]
}
with open('models/elbow_data.json', 'w') as f:
    json.dump(elbow_data, f)

# ==================== 5. BACKPROPAGATION (Manual NumPy) ====================
print("\n[5] Training Backpropagation (Manual NumPy)...")



bp_nn = BackpropNN(layer_sizes=[X_train.shape[1], 32, 16, 1], lr=0.005)
bp_nn.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=32)

y_pred_bp = bp_nn.predict(X_test)
y_pred_bp_orig = scaler_y.inverse_transform(y_pred_bp.reshape(-1, 1)).flatten()

mae_bp = mean_absolute_error(y_test_orig, y_pred_bp_orig)
rmse_bp = np.sqrt(mean_squared_error(y_test_orig, y_pred_bp_orig))
r2_bp = r2_score(y_test_orig, y_pred_bp_orig)

results['Backpropagation'] = {'MAE': round(mae_bp, 2), 'RMSE': round(rmse_bp, 2), 'R2': round(r2_bp, 4)}
print(f"  MAE: {mae_bp:.2f} | RMSE: {rmse_bp:.2f} | R2: {r2_bp:.4f}")
joblib.dump(bp_nn, 'models/backprop.pkl')

bp_loss = {
    'train_loss': bp_nn.loss_history,
    'val_loss': bp_nn.val_loss_history
}
with open('models/bp_loss.json', 'w') as f:
    json.dump(bp_loss, f)

# ==================== SAVE SCALERS & ENCODERS ====================
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
joblib.dump(le_genre, 'models/le_genre.pkl')
joblib.dump(le_platform, 'models/le_platform.pkl')
joblib.dump(le_dev, 'models/le_dev.pkl')
joblib.dump(le_esports, 'models/le_esports.pkl')
joblib.dump(le_trend, 'models/le_trend.pkl')

with open('models/results.json', 'w') as f:
    json.dump(results, f)

with open('models/feature_info.json', 'w') as f:
    json.dump({
        'genres': list(le_genre.classes_),
        'platforms': list(le_platform.classes_),
        'developers': list(le_dev.classes_),
        'esports': list(le_esports.classes_),
        'trends': list(le_trend.classes_),
        'feature_cols': feature_cols
    }, f)

print("\n✅ All models trained and saved!")
print("\n=== RESULTS SUMMARY ===")
for model, metrics in results.items():
    print(f"{model}: {metrics}")
