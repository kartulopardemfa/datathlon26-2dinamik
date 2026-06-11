"""Datathon 2026 - career_success_score prediction.

Pipeline:
1. Tabular feature engineering (skill aggregates, role-skill match, interactions).
2. Text features from mentor_feedback_text:
   - TF-IDF (word 1-3gram + char_wb 3-5gram) -> Ridge OOF prediction
   - Sentence-transformer embeddings -> Ridge OOF prediction
   - TF-IDF SVD components + embedding PCA components as direct features.
3. LightGBM / XGBoost / CatBoost, 5-fold CV, 2 seeds each.
4. OOF-optimized blend weights, predictions clipped to [0, 100].
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from scipy.sparse import hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, '.')
from fe import engineer

SEED = 42
N_FOLDS = 5
CAT_COLS = ['department', 'university_tier', 'target_role', 'hobby',
            'preferred_social_media_platform']
DROP = ['student_id', 'career_success_score', 'mentor_feedback_text']

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
y = tr['career_success_score'].values
kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(tr))

# ---------- text features ----------
txt_tr = tr['mentor_feedback_text'].fillna('')
txt_te = te['mentor_feedback_text'].fillna('')
all_txt = pd.concat([txt_tr, txt_te])

tw = TfidfVectorizer(max_features=80000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
tc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
                     sublinear_tf=True, min_df=2)
A = hstack([tw.fit_transform(all_txt), tc.fit_transform(all_txt)]).tocsr()
Atr, Ate = A[:len(tr)], A[len(tr):]

tfidf_oof = np.zeros(len(tr)); tfidf_te = np.zeros(len(te))
for tr_i, va_i in folds:
    r = Ridge(alpha=2.0)
    r.fit(Atr[tr_i], y[tr_i])
    tfidf_oof[va_i] = r.predict(Atr[va_i])
    tfidf_te += r.predict(Ate) / N_FOLDS
print('tfidf ridge MSE:', mean_squared_error(y, tfidf_oof))

E_tr = np.load('cache_emb_tr.npy')
E_te = np.load('cache_emb_te.npy')
emb_oof = np.zeros(len(tr)); emb_te = np.zeros(len(te))
for tr_i, va_i in folds:
    r = Ridge(alpha=10.0)
    r.fit(E_tr[tr_i], y[tr_i])
    emb_oof[va_i] = r.predict(E_tr[va_i])
    emb_te += r.predict(E_te) / N_FOLDS
print('emb ridge MSE:', mean_squared_error(y, emb_oof))

svd = TruncatedSVD(n_components=64, random_state=SEED)
S = svd.fit_transform(A)
Str, Ste = S[:len(tr)], S[len(tr):]

pca = PCA(n_components=32, random_state=SEED)
P_all = pca.fit_transform(np.vstack([E_tr, E_te]))
Ptr, Pte = P_all[:len(tr)], P_all[len(tr):]

# ---------- tabular matrix ----------
def build(df, S_, P_, tf_pred, em_pred):
    f = engineer(df).drop(columns=[c for c in DROP if c in df.columns]).reset_index(drop=True)
    f = pd.concat([f,
                   pd.DataFrame(S_, columns=[f'svd_{i}' for i in range(S_.shape[1])]),
                   pd.DataFrame(P_, columns=[f'emb_{i}' for i in range(P_.shape[1])])], axis=1)
    f['text_pred_tfidf'] = tf_pred
    f['text_pred_emb'] = em_pred
    return f

X = build(tr, Str, Ptr, tfidf_oof, emb_oof)
Xte = build(te, Ste, Pte, tfidf_te, emb_te)

# ---------- models ----------
def run_lgb(seed):
    oof = np.zeros(len(X)); tep = np.zeros(len(Xte))
    Xl, Xtl = X.copy(), Xte.copy()
    for c in CAT_COLS:
        Xl[c] = Xl[c].astype('category')
        Xtl[c] = Xtl[c].astype('category').cat.set_categories(Xl[c].cat.categories)
    for tr_i, va_i in folds:
        m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.02, num_leaves=63,
                              colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                              min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=seed, verbose=-1)
        m.fit(Xl.iloc[tr_i], y[tr_i], eval_set=[(Xl.iloc[va_i], y[va_i])],
              callbacks=[lgb.early_stopping(300, verbose=False)])
        oof[va_i] = m.predict(Xl.iloc[va_i])
        tep += m.predict(Xtl) / N_FOLDS
    return oof, tep

def run_xgb(seed):
    oof = np.zeros(len(X)); tep = np.zeros(len(Xte))
    Xx, Xtx = X.copy(), Xte.copy()
    for c in CAT_COLS:
        Xx[c] = Xx[c].astype('category')
        Xtx[c] = Xtx[c].astype('category').cat.set_categories(Xx[c].cat.categories)
    for tr_i, va_i in folds:
        m = xgb.XGBRegressor(n_estimators=6000, learning_rate=0.02, max_depth=6,
                             colsample_bytree=0.7, subsample=0.8, min_child_weight=5,
                             reg_alpha=0.1, reg_lambda=1.0, enable_categorical=True,
                             tree_method='hist', early_stopping_rounds=300,
                             random_state=seed)
        m.fit(Xx.iloc[tr_i], y[tr_i], eval_set=[(Xx.iloc[va_i], y[va_i])], verbose=False)
        oof[va_i] = m.predict(Xx.iloc[va_i])
        tep += m.predict(Xtx) / N_FOLDS
    return oof, tep

def run_cat(seed):
    oof = np.zeros(len(X)); tep = np.zeros(len(Xte))
    Xc, Xtc = X.copy(), Xte.copy()
    for c in CAT_COLS:
        Xc[c] = Xc[c].astype(str)
        Xtc[c] = Xtc[c].astype(str)
    for tr_i, va_i in folds:
        m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6,
                              l2_leaf_reg=3, cat_features=CAT_COLS,
                              early_stopping_rounds=400, random_seed=seed, verbose=False)
        m.fit(Xc.iloc[tr_i], y[tr_i], eval_set=(Xc.iloc[va_i], y[va_i]))
        oof[va_i] = m.predict(Xc.iloc[va_i])
        tep += m.predict(Xtc) / N_FOLDS
    return oof, tep

oofs, teps, names = [], [], []
for seed in (42, 2026):
    for name, fn in (('lgb', run_lgb), ('xgb', run_xgb), ('cat', run_cat)):
        o, t = fn(seed)
        oofs.append(o); teps.append(t); names.append(f'{name}_{seed}')
        print(f'{name}_{seed} MSE: {mean_squared_error(y, np.clip(o, 0, 100)):.4f}')

O = np.vstack(oofs).T
T = np.vstack(teps).T
np.save('cache_O.npy', O); np.save('cache_T.npy', T)

def loss(w):
    w = np.abs(w); w = w / w.sum()
    return mean_squared_error(y, np.clip(O @ w, 0, 100))

best = None
for init in (np.ones(O.shape[1]), np.random.RandomState(0).rand(O.shape[1])):
    res = minimize(loss, init, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
    if best is None or res.fun < best.fun:
        best = res
w = np.abs(best.x); w = w / w.sum()
print('weights:', dict(zip(names, np.round(w, 4))))
print('BLEND CV MSE:', loss(best.x))

pred = np.clip(T @ w, 0, 100)
sub = pd.DataFrame({'student_id': te['student_id'], 'career_success_score': pred})
sub.to_csv('submission.csv', index=False)
np.save('cache_blend_oof.npy', np.clip(O @ w, 0, 100))
print('submission.csv written', sub.shape)
