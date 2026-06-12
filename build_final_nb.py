import nbformat as nbf, re
src = open('build_kaggle_nb2.py').read()
m = re.findall(r"C\.append\(nbf\.v4\.new_code_cell\(r'''(.*?)'''\)\)", src, flags=re.S)
setup, bert, textfeat, fe, models, blend = m
src3 = open('build_kaggle_nb3.py').read()
fusion = re.search(r"fusion = r'''(.*?)'''", src3, flags=re.S).group(1)

# fusion x2 seeds
fusion2 = fusion.replace("torch.manual_seed(SEED)", "pass")
fusion2 = fusion2.replace(
"""te_dl = DataLoader(DS(tx_te, TAB_te), batch_size=128)
fus_oof = np.zeros(len(tr)); fus_te = np.zeros(len(te))
for fold,(tr_i,va_i) in enumerate(folds):""",
"""te_dl = DataLoader(DS(tx_te, TAB_te), batch_size=128)
FUS = {}
for fseed in (42, 7):
  torch.manual_seed(fseed)
  fus_oof = np.zeros(len(tr)); fus_te = np.zeros(len(te))
  for fold,(tr_i,va_i) in enumerate(folds):""")
fusion2 = re.sub(r"\n(    model = Fusion)", r"\n    \1", fusion2)  # noop guard
# reindent the per-fold body under the new loop: simpler — replace tail block
fusion2 = fusion2.replace(
"""RESULTS = globals().get('RESULTS', {})""",
"""  FUS[fseed] = (fus_oof.copy(), fus_te.copy())
  print('seed', fseed, 'OOF MSE %.2f' % mean_squared_error(y, np.clip(fus_oof,0,100)), flush=True)
RESULTS = globals().get('RESULTS', {})""")
# fix indentation of inner fold loop body (add 2 spaces to lines between markers)
lines = fusion2.split('\n')
out=[]; inside=False
for i,l in enumerate(lines):
    if l.startswith('  for fold,(tr_i,va_i)'): inside=True; out.append(l); continue
    if inside:
        if l.startswith('  FUS[fseed]'): inside=False; out.append(l); continue
        out.append('  '+l if l.strip() else l)
    else:
        out.append(l)
fusion2='\n'.join(out)
fusion2 = fusion2.replace("""RESULTS['fusion_mse'] = mean_squared_error(y, np.clip(fus_oof,0,100))
RESULTS['fusion_lb'] = lb_est(np.clip(fus_oof,0,100))
RESULTS['fusion_corr'] = np.corrcoef(fus_oof, y)[0,1]
print('FUZYON OOF: uniform=%.3f LB_est=%.3f corr=%.4f' % (RESULTS['fusion_mse'], RESULTS['fusion_lb'], RESULTS['fusion_corr']))
np.save('fusion_oof.npy', fus_oof); np.save('fusion_te.npy', fus_te)
bert_oof, bert_te = fus_oof, fus_te   # asagidaki hucreler bert_* adini kullaniyor
RESULTS['bert_mse']=RESULTS['fusion_mse']; RESULTS['bert_corr']=RESULTS['fusion_corr']""",
"""fus_oof = (FUS[42][0]+FUS[7][0])/2; fus_te = (FUS[42][1]+FUS[7][1])/2
RESULTS['fusion_mse'] = mean_squared_error(y, np.clip(fus_oof,0,100))
RESULTS['fusion_lb'] = lb_est(np.clip(fus_oof,0,100))
RESULTS['fusion_corr'] = np.corrcoef(fus_oof, y)[0,1]
print('FUZYON(2seed) OOF: uniform=%.3f LB_est=%.3f corr=%.4f' % (RESULTS['fusion_mse'], RESULTS['fusion_lb'], RESULTS['fusion_corr']))
np.save('fusion_oof.npy', fus_oof); np.save('fusion_te.npy', fus_te)
bert_oof, bert_te = fus_oof, fus_te
RESULTS['bert_mse']=RESULTS['fusion_mse']; RESULTS['bert_corr']=RESULTS['fusion_corr']""")

# persona feature into fe cell
fe2 = fe.replace("""X=build(tr,Str_,Ptr_,imp_tr,tfidf_oof,emb_oof,bert_oof,knn_tr)""",
"""def persona_feats(texts):
    tl = pd.Series(texts).fillna('').str.lower()
    return pd.DataFrame({'p_siz': (tl.str.startswith('siz')|tl.str.contains('skorlarınız|projeleriniz|becerileriniz')).astype(int),
                         'p_ogrenci': tl.str.contains('bu öğrenci').astype(int)})
X=build(tr,Str_,Ptr_,imp_tr,tfidf_oof,emb_oof,bert_oof,knn_tr)""")
fe2 = fe2.replace("""Xte=build(te,Ste_,Pte_,imp_te,tfidf_te,emb_te,bert_te,knn_te)""",
"""Xte=build(te,Ste_,Pte_,imp_te,tfidf_te,emb_te,bert_te,knn_te)
X = pd.concat([X, persona_feats(tx_tr)], axis=1); Xte = pd.concat([Xte, persona_feats(tx_te)], axis=1)""")

# optuna cell
optuna_cell = r'''
# Optuna ile CatBoost ayar taramasi (GPU, ~45 dk)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
Xc = X.copy()
for c in CAT_COLS: Xc[c]=Xc[c].astype(str)
f3 = folds[:3]
def objective(t):
    params = dict(iterations=4000, learning_rate=t.suggest_float('lr',0.02,0.08,log=True),
                  depth=t.suggest_int('depth',5,8), l2_leaf_reg=t.suggest_float('l2',1,10,log=True),
                  random_strength=t.suggest_float('rs',0.1,3,log=True),
                  bagging_temperature=t.suggest_float('bt',0,2),
                  early_stopping_rounds=300, random_seed=42, verbose=False,
                  cat_features=CAT_COLS, task_type='GPU' if torch.cuda.is_available() else 'CPU')
    errs=[]
    for tr_i,va_i in f3:
        m=CatBoostRegressor(**params)
        try: m.fit(Xc.iloc[tr_i],y[tr_i],eval_set=(Xc.iloc[va_i],y[va_i]))
        except Exception:
            params.pop('task_type',None); m=CatBoostRegressor(**params)
            m.fit(Xc.iloc[tr_i],y[tr_i],eval_set=(Xc.iloc[va_i],y[va_i]))
        p=np.clip(m.predict(Xc.iloc[va_i]),0,100)
        errs.append(np.average((y[va_i]-p)**2, weights=iw[va_i]))
    return float(np.mean(errs))
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)
BEST = study.best_params
print('OPTUNA best:', BEST, 'val:', round(study.best_value,3))
'''

# tuned models cell: use BEST in run_cat
models2 = models.replace("('cat_42',lambda: run_cat(42)),('catNT_42',lambda: run_cat(42,native_text=True)),\n      ('cat_2026',lambda: run_cat(2026)),('lgb_42',lambda: run_lgb(42)),('xgb_42',lambda: run_xgb(42))",
 "('cat_42',lambda: run_cat(42)),('cat_2026',lambda: run_cat(2026)),('cat_7',lambda: run_cat(7)),('lgb_42',lambda: run_lgb(42)),('lgb_2026',lambda: run_lgb(2026)),('xgb_42',lambda: run_xgb(42)),('xgb_2026',lambda: run_xgb(2026))")
models2 = models2.replace("m=CatBoostRegressor(iterations=10000,learning_rate=0.03,depth=6,l2_leaf_reg=3,",
 "m=CatBoostRegressor(iterations=10000,learning_rate=BEST.get('lr',0.03),depth=BEST.get('depth',6),l2_leaf_reg=BEST.get('l2',3),random_strength=BEST.get('rs',1),bagging_temperature=BEST.get('bt',1),")
models2 = models2.replace("m=CatBoostRegressor(iterations=10000,learning_rate=0.03,depth=6,l2_leaf_reg=3,\n                                cat_features=CAT_COLS,early_stopping_rounds=400,random_seed=seed,\n                                verbose=False,**kw)",
 "m=CatBoostRegressor(iterations=10000,learning_rate=BEST.get('lr',0.03),depth=BEST.get('depth',6),l2_leaf_reg=BEST.get('l2',3),random_strength=BEST.get('rs',1),bagging_temperature=BEST.get('bt',1),cat_features=CAT_COLS,early_stopping_rounds=400,random_seed=seed,verbose=False,**kw)")
models2 = models2.replace("for nm in ['cat_42','catNT_42','cat_2026','lgb_42','xgb_42']:", "pass")

blend2 = blend.replace("TANI RAPORU v2", "FINAL TANI RAPORU")
blend2 = blend2.replace("""for nm in ['cat_42','catNT_42','cat_2026','lgb_42','xgb_42']:
    print('%s: LB_est=%.3f' % (nm, RESULTS[nm]))""",
"""for nm in [k for k in RESULTS if k.startswith(('cat_','lgb_','xgb_'))]:
    print('%s: LB_est=%.3f' % (nm, RESULTS[nm]))
print('FUZYON: LB_est=%.3f corr=%.4f' % (RESULTS['fusion_lb'], RESULTS['fusion_corr']))""")
blend2 = blend2.replace("print('BERT: MSE=%.2f corr=%.4f' % (RESULTS['bert_mse'], RESULTS['bert_corr']))", "")

nb = nbf.v4.new_notebook()
nb['cells'] = [nbf.v4.new_markdown_cell("""# Datathon 2026 — FINAL Notebook (Füzyon ×2 + Optuna + Mega Harman)

**Kurulum:** GPU + Internet ON + yarışma verisi → Run All (~3.5-4 saat)
Üretilenler: `submission.csv` + `O.npy`/`T.npy` (bana yükleyin) + FINAL TANI RAPORU."""),
 nbf.v4.new_code_cell(setup),
 nbf.v4.new_code_cell(fusion2),
 nbf.v4.new_code_cell(textfeat),
 nbf.v4.new_code_cell(fe2),
 nbf.v4.new_code_cell(optuna_cell),
 nbf.v4.new_code_cell(models2),
 nbf.v4.new_code_cell(blend2)]
nbf.write(nb, 'kaggle_final.ipynb')
print('written')
