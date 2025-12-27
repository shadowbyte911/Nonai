import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5) # On analyse les 5 premières secondes
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Erreur sur {file_path}: {e}")
        return None

# Préparation des données
X = [] # Caractéristiques
y = [] # Labels (0 pour humain, 1 pour IA)

# On boucle sur les dossiers pour l'entraînement
for label, folder in enumerate(['data/humain', 'data/ai']):
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            feat = extract_features(os.path.join(folder, file))
            if feat is not None:
                X.append(feat)
                y.append(label)

# Entraînement du modèle
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

# Sauvegarde du modèle pour l'utiliser plus tard
joblib.dump(clf, 'detecteur_audio.joblib')
print("Modèle entraîné et sauvegardé !")