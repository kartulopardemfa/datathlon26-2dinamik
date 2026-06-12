"""Test 3 hypotheses: CatBoost native text, sentiment-gated skills, kNN-target."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, '.')
from fe import engineer
from aspect_fe import aspect_features, ASPECTS

CAT_COLS = ['department', 'university_tier', 'target_role', 'hobby',
            'preferred_social_media_platform']
DROP = ['student_id', 'career_success_score', 'mentor_feedback_text']

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
y = tr['career_success_score'].values
folds = list(KFold(5, shuffle=True, random_state=42).split(tr))
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_ / trp).values
def lb_est(p):
    return np.average((y - p) ** 2, weights=iw)

Str = np.load('cache_svd_tr.npy')
toof = np.load('cache_toof2.npy'); eoof = np.load('cache_eoof.npy')
imp_tr = pd.read_csv('cache_impflags_train.csv')

base = engineer(tr).drop(columns=DROP).reset_index(drop=True)
base = pd.concat([base, pd.DataFrame(Str, columns=[f'svd_{i}' for i in range(64)]),
                  imp_tr], axis=1)
base['text_pred'] = toof; base['text_pred_emb'] = eoof

# ---- H1: CatBoost native text_features (raw text + tabular WITHOUT svd/meta) ----
X1 = engineer(tr).drop(columns=['student_id', 'career_success_score']).reset_index(drop=True)
X1 = pd.concat([X1, imp_tr], axis=1)
X1['mentor_feedback_text'] = tr['mentor_feedback_text'].fillna('')
for c in CAT_COLS: X1[c] = X1[c].astype(str)
oof = np.zeros(len(X1))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT_COLS, text_features=['mentor_feedback_text'],
                          early_stopping_rounds=400, random_seed=42, verbose=False)
    m.fit(X1.iloc[tr_i], y[tr_i], eval_set=(X1.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X1.iloc[va_i])
oc = np.clip(oof, 0, 100)
print('H1 CatBoost native text: uniform=%.3f LB_est=%.3f' % (mean_squared_error(y, oc), lb_est(oc)), flush=True)
np.save('cache_oof_h1.npy', oof)

# ---- H2: sentiment-gated skill features ----
asp = pd.read_csv('cache_aspects_tr.csv')
GATE = {'kodlama': 'coding_score', 'problem': 'problem_solving_score',
        'veri_yapi': 'data_structures_score', 'sql': 'sql_score',
        'ml': 'machine_learning_score', 'backend': 'backend_score',
        'frontend': 'frontend_score', 'cloud': 'cloud_score', 'devops': 'devops_score',
        'proje': 'project_quality_score', 'iletisim': 'communication_score',
        'takim': 'teamwork_score', 'liderlik': 'leadership_score',
        'mulakat': 'technical_interview_score', 'portfoy': 'portfolio_score'}
X2 = base.copy()
for a, col in GATE.items():
    s = asp[f'asp_{a}'].values
    v = tr[col].fillna(tr[col].median()).values
    X2[f'gate_{a}'] = s * v                    # sentiment x value
    X2[f'mention_{a}'] = (s != 0) * v          # mentioned-at-all x value
X2['gated_sum'] = sum(asp[f'asp_{a}'].values * tr[c].fillna(tr[c].median()).values
                      for a, c in GATE.items())
for c in CAT_COLS: X2[c] = X2[c].astype(str)
oof = np.zeros(len(X2))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT_COLS, early_stopping_rounds=400,
                          random_seed=42, verbose=False)
    m.fit(X2.iloc[tr_i], y[tr_i], eval_set=(X2.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X2.iloc[va_i])
oc = np.clip(oof, 0, 100)
print('H2 sentiment-gated: uniform=%.3f LB_est=%.3f' % (mean_squared_error(y, oc), lb_est(oc)), flush=True)
np.save('cache_oof_h2.npy', oof)

# ---- H3: kNN-target features over embedding space (OOF-clean) ----
from sklearn.neighbors import NearestNeighbors
E = np.load('cache_emb_tr.npy')
E = E / np.linalg.norm(E, axis=1, keepdims=True)
knn_feats = np.zeros((len(tr), 4))
for tr_i, va_i in folds:
    nn = NearestNeighbors(n_neighbors=20).fit(E[tr_i])
    d, idx = nn.kneighbors(E[va_i])
    ny = y[tr_i][idx]
    knn_feats[va_i, 0] = ny[:, :5].mean(1)
    knn_feats[va_i, 1] = ny[:, :20].mean(1)
    knn_feats[va_i, 2] = ny[:, :20].std(1)
    knn_feats[va_i, 3] = d[:, :5].mean(1)
X3 = base.copy()
for j, n in enumerate(['knn5_y', 'knn20_y', 'knn20_std', 'knn5_dist']):
    X3[n] = knn_feats[:, j]
for c in CAT_COLS: X3[c] = X3[c].astype(str)
oof = np.zeros(len(X3))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT_COLS, early_stopping_rounds=400,
                          random_seed=42, verbose=False)
    m.fit(X3.iloc[tr_i], y[tr_i], eval_set=(X3.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X3.iloc[va_i])
oc = np.clip(oof, 0, 100)
print('H3 kNN-target: uniform=%.3f LB_est=%.3f' % (mean_squared_error(y, oc), lb_est(oc)), flush=True)
np.save('cache_oof_h3.npy', oof)
print('REFERANS: impflags CatBoost LB_est=86.868')
