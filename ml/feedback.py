import numpy as np

def get_nba_feature_stats(nba_feature_matrix):
    return np.mean(nba_feature_matrix, axis=0), np.std(nba_feature_matrix, axis=0)

def generate_feedback(user_features, nba_mean, nba_std, feature_names):
    feedback = []
    for i, (uf, nm, ns, name) in enumerate(zip(user_features, nba_mean, nba_std, feature_names)):
        if abs(uf - nm) > ns:  
            if uf < nm:
                feedback.append(f"Increase your {name} (yours: {uf:.1f}, Professional avg: {nm:.1f})")
            else:
                feedback.append(f"Decrease your {name} (yours: {uf:.1f}, Professional avg: {nm:.1f})")
    return feedback