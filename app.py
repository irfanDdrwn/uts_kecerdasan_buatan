from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os
from backprop_model import BackpropNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')

scaler_X  = joblib.load(f'{MODEL_DIR}/scaler_X.pkl')
scaler_y  = joblib.load(f'{MODEL_DIR}/scaler_y.pkl')
scaler_km = joblib.load(f'{MODEL_DIR}/scaler_km.pkl')
le_genre    = joblib.load(f'{MODEL_DIR}/le_genre.pkl')
le_platform = joblib.load(f'{MODEL_DIR}/le_platform.pkl')
le_dev      = joblib.load(f'{MODEL_DIR}/le_dev.pkl')
le_esports  = joblib.load(f'{MODEL_DIR}/le_esports.pkl')
le_trend    = joblib.load(f'{MODEL_DIR}/le_trend.pkl')
lr_model  = joblib.load(f'{MODEL_DIR}/linear_regression.pkl')
bp_model  = joblib.load(f'{MODEL_DIR}/backprop.pkl')
km_model  = joblib.load(f'{MODEL_DIR}/kmeans.pkl')

_ann_model = None
_rnn_model = None

def get_ann():
    global _ann_model
    if _ann_model is None:
        _ann_model = keras.models.load_model(f'{MODEL_DIR}/ann_model.h5', compile=False)
    return _ann_model

def get_rnn():
    global _rnn_model
    if _rnn_model is None:
        _rnn_model = keras.models.load_model(f'{MODEL_DIR}/rnn_model.h5', compile=False)
    return _rnn_model

with open(f'{MODEL_DIR}/results.json') as f:
    model_results = json.load(f)
with open(f'{MODEL_DIR}/feature_info.json') as f:
    feat_info = json.load(f)
with open(f'{MODEL_DIR}/ann_loss.json') as f:
    ann_loss = json.load(f)
with open(f'{MODEL_DIR}/rnn_loss.json') as f:
    rnn_loss = json.load(f)
with open(f'{MODEL_DIR}/bp_loss.json') as f:
    bp_loss = json.load(f)
with open(f'{MODEL_DIR}/elbow_data.json') as f:
    elbow_data = json.load(f)

CLUSTER_LABELS = {0: 'Budget Indie', 1: 'Mainstream Hit', 2: 'Blockbuster', 3: 'Niche Premium'}

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html', feat_info=feat_info)

@app.route('/comparison')
def comparison():
    return render_template('comparison.html',
                           results=model_results,
                           ann_loss=ann_loss,
                           rnn_loss=rnn_loss,
                           bp_loss=bp_loss,
                           elbow_data=elbow_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    try:
        release_year   = float(data['release_year'])
        players        = float(data['players'])
        peak_players   = float(data['peak_players'])
        metacritic     = float(data['metacritic'])
        genre          = data['genre']
        platform       = data['platform']
        developer      = data['developer']
        esports        = data['esports']
        trend          = data['trend']
        model_choice   = data['model']

        genre_enc    = le_genre.transform([genre])[0]
        platform_enc = le_platform.transform([platform])[0]
        dev_enc      = le_dev.transform([developer])[0]
        esports_enc  = le_esports.transform([esports])[0]
        trend_enc    = le_trend.transform([trend])[0]

        X_raw = np.array([[release_year, players, peak_players, metacritic,
                           genre_enc, platform_enc, dev_enc, esports_enc, trend_enc]])
        X_sc  = scaler_X.transform(X_raw)

        if model_choice == 'linear_regression':
            pred_sc = lr_model.predict(X_sc)
        elif model_choice == 'ann':
            pred_sc = get_ann().predict(X_sc, verbose=0).flatten()
        elif model_choice == 'rnn':
            X_rnn = X_sc.reshape(1, 1, X_sc.shape[1])
            pred_sc = get_rnn().predict(X_rnn, verbose=0).flatten()
        elif model_choice == 'backprop':
            pred_sc = bp_model.predict(X_sc)
        else:
            return jsonify({'error': 'Model not found'}), 400

        pred_orig = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()[0]
        pred_orig = max(0, pred_orig)

        # K-Means cluster
        X_km_raw = np.array([[pred_orig, players, peak_players, metacritic]])
        X_km_sc  = scaler_km.transform(X_km_raw)
        cluster  = km_model.predict(X_km_sc)[0]
        cluster_label = CLUSTER_LABELS.get(int(cluster), f'Cluster {cluster}')

        return jsonify({
            'revenue': round(float(pred_orig), 2),
            'cluster': int(cluster),
            'cluster_label': cluster_label,
            'model_used': model_choice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
