import joblib
import librosa
import numpy as np

# 1. On REDÉFINIT la fonction d'extraction (indispensable)
def extract_features(file_path):
    try:
        # On charge l'audio (sr=None garde la qualité originale)
        y, sr = librosa.load(file_path, duration=5) 
        # On extrait les MFCC (les empreintes de la voix)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # On calcule la moyenne pour avoir un vecteur de taille fixe
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None

# 2. La fonction de détection
def detecter_audio(file_path):
    try:
        # Charger le modèle entraîné précédemment
        model = joblib.load('detecteur_audio.joblib')
        
        # Extraire les caractéristiques du nouveau fichier
        features = extract_features(file_path)
        
        if features is not None:
            # Reformater pour Scikit-Learn (1 échantillon, N caractéristiques)
            features = features.reshape(1, -1)
            
            # Faire la prédiction
            prediction = model.predict(features)
            proba = model.predict_proba(features)
            
            categories = ["Humain", "IA"]
            resultat = categories[prediction[0]]
            confiance = proba[0][prediction[0]] * 100
            
            print(f"--- Analyse terminée ---")
            print(f"Fichier : {file_path}")
            print(f"Résultat : {resultat}")
            print(f"Confiance : {confiance:.2f}%")
        
    except FileNotFoundError:
        print("Erreur : Le fichier 'detecteur_audio.joblib' est introuvable. Avez-vous lancé le script d'entraînement ?")

# 3. Exécution
if __name__ == "__main__":
    nom_fichier = 'Tech that refuses to die  6 Minute English.wav'
    detecter_audio(nom_fichier)