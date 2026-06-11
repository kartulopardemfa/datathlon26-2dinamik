"""Final ensemble v2: engineered features + imputation flags + 3 text meta-features
(TF-IDF ridge, sentence-embedding ridge, fine-tuned MiniLM) + SVD/PCA text components.
LGB/XGB/CatBoost x 2 seeds + MLP, blend optimized on year-weighted (LB-matched) MSE.
Requires caches: cache_svd_*, cache_emb_*, cache_toof2/tpred2, cache_eoof/epred,
cache_ft_oof/cache_ft_te, cache_impflags_*.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.decomposition import PCA
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

tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_ / trp).values
def lb_est(p):
    return np.average((y - p) ** 2, weights=iw)

# ---------- cached text features ----------
Str, Ste = np.load('cache_svd_tr.npy'), np.load('cache_svd_te.npy')
E_tr, E_te = np.load('cache_emb_tr.npy'), np.load('cache_emb_te.npy')
P_all = PCA(n_components=32, random_state=SEED).fit_transform(np.vstack([E_tr, E_te]))
Ptr, Pte = P_all[:len(tr)], P_all[len(tr):]
toof, tpred = np.load('cache_toof2.npy'), np.load('cache_tpred2.npy')
eoof, epred = np.load('cache_eoof.npy'), np.load('cache_epred.npy')
ft_oof, ft_te = np.load('cache_ft_oof.npy'), np.load('cache_ft_te.npy')
print('FT text standalone MSE:', mean_squared_error(y, ft_oof))
imp_tr = pd.read_csv('cache_impflags_train.csv')
imp_te = pd.read_csv('cache_impflags_test.csv')

def build(df, S_, P_, impf, tf, em, ft):
    f = engineer(df).drop(columns=[c for c in DROP if c in df.columns]).reset_index(drop=True)
    f = pd.concat([f,
                   pd.DataFrame(S_, columns=[f'svd_{i}' for i in range(S_.shape[1])]),
                   pd.DataFrame(P_, columns=[f'emb_{i}' for i in range(P_.shape[1])]),
                   impf.reset_index(drop=True)], axis=1)
    f['text_pred_tfidf'] = tf
    f['text_pred_emb'] = em
    f['text_pred_ft'] = ft
    return f

X = build(tr, Str, Ptr, imp_tr, toof, eoof, ft_oof)
Xte = build(te, Ste, Pte, imp_te, tpred, epred, ft_te)

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

def run_mlp(seed):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    C_tr = ohe.fit_transform(X[CAT_COLS]); C_te = ohe.transform(Xte[CAT_COLS])
    imp = SimpleImputer(strategy='median')
    N_tr = imp.fit_transform(X.drop(columns=CAT_COLS))
    N_te = imp.transform(Xte.drop(columns=CAT_COLS))
    sc = StandardScaler()
    Mtr = sc.fit_transform(np.hstack([N_tr, C_tr]))
    Mte = sc.transform(np.hstack([N_te, C_te]))
    oof = np.zeros(len(X)); tep = np.zeros(len(Xte))
    for tr_i, va_i in folds:
        m = MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                         learning_rate_init=1e-3, batch_size=256, max_iter=200,
                         early_stopping=True, n_iter_no_change=15, random_state=seed)
        m.fit(Mtr[tr_i], y[tr_i])
        oof[va_i] = m.predict(Mtr[va_i])
        tep += m.predict(Mte) / N_FOLDS
    return oof, tep

oofs, teps, names = [], [], []
for seed in (42, 2026):
    for name, fn in (('lgb', run_lgb), ('xgb', run_xgb), ('cat', run_cat)):
        o, t = fn(seed)
        oofs.append(o); teps.append(t); names.append(f'{name}_{seed}')
        print(f'{name}_{seed}: uniform={mean_squared_error(y, np.clip(o,0,100)):.3f} '
              f'LB_est={lb_est(np.clip(o,0,100)):.3f}', flush=True)
o, t = run_mlp(42)
oofs.append(o); teps.append(t); names.append('mlp_42')
print(f'mlp_42: uniform={mean_squared_error(y, np.clip(o,0,100)):.3f} '
      f'LB_est={lb_est(np.clip(o,0,100)):.3f}', flush=True)

# raw text predictions as extra blend components
oofs += [toof, eoof, ft_oof]
teps += [tpred, epred, ft_te]
names += ['tfidf', 'emb', 'ft']

O = np.vstack(oofs).T
T = np.vstack(teps).T
np.save('cache_O2.npy', O); np.save('cache_T2.npy', T)

def opt(M, yt, w_eval):
    def loss(w):
        w = np.abs(w); w = w / w.sum()
        return np.average((yt - np.clip(M @ w, 0, 100)) ** 2, weights=w_eval)
    best = None
    for s in range(5):
        r = minimize(loss, np.random.RandomState(s).rand(M.shape[1]) + 0.5,
                     method='Nelder-Mead', options={'maxiter': 12000, 'fatol': 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    w = np.abs(best.x); w = w / w.sum()
    return w, best.fun

w_u, f_u = opt(O, y, None)
print('uniform-opt blend: LB_est=%.3f' % lb_est(np.clip(O @ w_u, 0, 100)))
w_w, f_w = opt(O, y, iw)
print('weighted-opt blend: LB_est=%.3f' % f_w)
print('weights:', dict(zip(names, np.round(w_w, 3))))

w_final = w_w if f_w <= lb_est(np.clip(O @ w_u, 0, 100)) else w_u
pred = np.clip(T @ w_final, 0, 100)
sub = pd.DataFrame({'student_id': te['student_id'], 'career_success_score': pred})
sub.to_csv('submission_v2.csv', index=False)
np.save('cache_w_v2.npy', w_final)
print('submission_v2.csv written', sub.shape)
