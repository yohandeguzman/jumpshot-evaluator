import os
import pickle
import numpy as np
from ml.feature_extraction import extract_shot_features
from sklearn.ensemble import RandomForestClassifier

# Optional: for future feedback use
# from ml.feedback import get_nba_feature_stats

def load_shot_csv(path):
    data = np.loadtxt(path, delimiter=',')
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)  # ensure it's 2D even if only 1 frame
    return data

X, y = [], []
data_dir = 'data/csv_files'

for fname in os.listdir(data_dir):
    if fname.endswith('.csv'):
        shot = load_shot_csv(os.path.join(data_dir, fname))
        
        if shot.shape[0] < 5:  # skip if fewer than 5 frames
            print(f"⚠️ Skipping {fname} — too short ({shot.shape[0]} frame(s))")
            continue
        
        features = extract_shot_features(shot)
        X.append(features)
        y.append(1)  # all NBA shots are "good form"

X = np.array(X)
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Save the trained model
with open('ml/model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save NBA feature matrix for feedback comparison later
np.save('ml/nba_features.npy', X)

print(f"✅ Trained model on {len(X)} NBA shots")
print("💾 Saved model to: ml/model.pkl")
print("📊 Saved feature matrix to: ml/nba_features.npy")
