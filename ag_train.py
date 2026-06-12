"""AutoGluon stacked ensemble on engineered features + text meta-features."""
import sys
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

sys.path.insert(0, '.')
from fe import engineer

DROP = ['student_id', 'career_success_score', 'mentor_feedback_text']

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
y = tr['career_success_score'].values
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_ / trp).values

Str, Ste = np.load('cache_svd_tr.npy'), np.load('cache_svd_te.npy')
toof, tpred = np.load('cache_toof2.npy'), np.load('cache_tpred2.npy')
eoof, epred = np.load('cache_eoof.npy'), np.load('cache_epred.npy')
ftoof, ftte = np.load('cache_ft_oof.npy'), np.load('cache_ft_te.npy')
imp_tr = pd.read_csv('cache_impflags_train.csv')
imp_te = pd.read_csv('cache_impflags_test.csv')

def build(df, S_, impf, tf, em, ft):
    f = engineer(df).drop(columns=[c for c in DROP if c in df.columns]).reset_index(drop=True)
    f = pd.concat([f, pd.DataFrame(S_, columns=[f'svd_{i}' for i in range(S_.shape[1])]),
                   impf.reset_index(drop=True)], axis=1)
    f['text_pred_tfidf'] = tf
    f['text_pred_emb'] = em
    f['text_pred_ft'] = ft
    return f

X = build(tr, Str, imp_tr, toof, eoof, ftoof)
Xte = build(te, Ste, imp_te, tpred, epred, ftte)
X['career_success_score'] = y

pred = TabularPredictor(label='career_success_score',
                        eval_metric='mean_squared_error',
                        path='ag_models')
pred.fit(X, presets='best_quality', time_limit=7200, num_cpus=4)

oof = pred.predict_oof() if hasattr(pred, 'predict_oof') else pred.get_oof_pred()
oof = np.clip(np.asarray(oof), 0, 100)
from sklearn.metrics import mean_squared_error
print('AG OOF uniform MSE:', mean_squared_error(y, oof), flush=True)
print('AG OOF LB est:', np.average((y - oof) ** 2, weights=iw), flush=True)
np.save('cache_ag_oof.npy', oof)

p_te = np.clip(pred.predict(Xte).values, 0, 100)
np.save('cache_ag_te.npy', p_te)
print('AG leaderboard:'); print(pred.leaderboard(silent=True).head(10).to_string(), flush=True)
print('done', flush=True)
