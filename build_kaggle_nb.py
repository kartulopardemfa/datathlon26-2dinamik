import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []

C.append(nbf.v4.new_markdown_cell("""# Datathon 2026 — GPU Pipeline (Türkçe BERT + Ensemble)

**Kullanım:**
1. Settings → Accelerator: **GPU (T4 x2 veya P100)**
2. Settings → Internet: **ON** (HuggingFace model indirmek için)
3. Add Data → yarışma veri setini ekleyin
4. Run All. Sonunda `submission.csv` üretilir ve tanı metrikleri yazdırılır.

Sonunda yazdırılan **TANI RAPORU** bloğunu geri gönderin."""))

C.append(nbf.v4.new_code_cell(r'''
import glob, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from scipy.sparse import hstack as sp_hstack
from scipy.optimize import minimize

# veri yolunu otomatik bul
cands = glob.glob('/kaggle/input/*/train.csv') + glob.glob('/kaggle/input/*/*/train.csv')
assert cands, 'train.csv bulunamadi - Add Data ile yarisma verisini ekleyin'
DATA = os.path.dirname(cands[0])
tr = pd.read_csv(f'{DATA}/train.csv', encoding='utf-8-sig')
te_path = f'{DATA}/test.csv' if os.path.exists(f'{DATA}/test.csv') else glob.glob(f'{DATA}/test*.csv')[0]
te = pd.read_csv(te_path, encoding='utf-8-sig')
y = tr['career_success_score'].values.astype(np.float64)
print(tr.shape, te.shape, '| GPU:', torch.cuda.is_available())

SEED = 42; N_FOLDS = 5
kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(tr))
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_/trp).values
def lb_est(p): return np.average((y-p)**2, weights=iw)   # LB skoru tahmini (dogrulandi: +-0.3)
'''))

C.append(nbf.v4.new_code_cell(r'''
# ---------- Turkce BERT fine-tune (GPU) ----------
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

MODEL='dbmdz/bert-base-turkish-cased'; MAXLEN=128; BS=32; EPOCHS=3
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
Y_MU, Y_SD = float(y.mean()), float(y.std())

class DS(Dataset):
    def __init__(self, texts, targets=None):
        self.enc = tok(list(texts), truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='np')
        self.t = targets
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k,v in self.enc.items()}
        if self.t is not None: d['target'] = torch.tensor(np.float32(self.t[i]))
        return d

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.bb.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask, **kw):
        o = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return self.head((o*m).sum(1)/m.sum(1)).squeeze(-1)

tx_tr = tr['mentor_feedback_text'].fillna('').values
tx_te = te['mentor_feedback_text'].fillna('').values
yz = (y - Y_MU)/Y_SD
te_dl = DataLoader(DS(tx_te), batch_size=128)
bert_oof = np.zeros(len(tr)); bert_te = np.zeros(len(te))
for fold,(tr_i,va_i) in enumerate(folds):
    model = Reg().to(DEV)
    opt = torch.optim.AdamW([{'params': model.bb.parameters(),'lr':2e-5},
                             {'params': model.head.parameters(),'lr':1e-3}])
    dl = DataLoader(DS(tx_tr[tr_i], yz[tr_i]), batch_size=BS, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(dl))
    lossf = nn.MSELoss(); model.train()
    for ep in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            p = model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV))
            loss = lossf(p, b['target'].to(DEV)); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        va_dl = DataLoader(DS(tx_tr[va_i]), batch_size=128)
        pv = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV)).cpu().numpy() for b in va_dl])
        bert_oof[va_i] = pv*Y_SD + Y_MU
        pt = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV)).cpu().numpy() for b in te_dl])
        bert_te += (pt*Y_SD + Y_MU)/N_FOLDS
    print(f'fold {fold}: val MSE {mean_squared_error(y[va_i], bert_oof[va_i]):.2f}', flush=True)
    del model; torch.cuda.empty_cache()
print('BERT OOF MSE: %.2f corr: %.4f' % (mean_squared_error(y, bert_oof), np.corrcoef(bert_oof,y)[0,1]))
np.save('bert_oof.npy', bert_oof); np.save('bert_te.npy', bert_te)
'''))

C.append(nbf.v4.new_code_cell(r'''
# ---------- TF-IDF ridge + sentence embedding (GPU) meta-ozellikleri ----------
all_txt = pd.concat([pd.Series(tx_tr), pd.Series(tx_te)])
tw = TfidfVectorizer(max_features=80000, ngram_range=(1,3), sublinear_tf=True, min_df=2)
tc = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=80000, sublinear_tf=True, min_df=2)
A = sp_hstack([tw.fit_transform(all_txt), tc.fit_transform(all_txt)]).tocsr()
Atr, Ate = A[:len(tr)], A[len(tr):]
tfidf_oof = np.zeros(len(tr)); tfidf_te = np.zeros(len(te))
for tr_i, va_i in folds:
    r = Ridge(alpha=2.0); r.fit(Atr[tr_i], y[tr_i])
    tfidf_oof[va_i] = r.predict(Atr[va_i]); tfidf_te += r.predict(Ate)/N_FOLDS
print('TFIDF ridge MSE:', mean_squared_error(y, tfidf_oof))

S = TruncatedSVD(n_components=64, random_state=SEED).fit_transform(A)
Str, Ste = S[:len(tr)], S[len(tr):]

from sentence_transformers import SentenceTransformer
st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device=DEV)
E_all = st.encode(list(all_txt), batch_size=256, show_progress_bar=False)
E_tr, E_te = E_all[:len(tr)], E_all[len(tr):]
emb_oof = np.zeros(len(tr)); emb_te = np.zeros(len(te))
for tr_i, va_i in folds:
    r = Ridge(alpha=10.0); r.fit(E_tr[tr_i], y[tr_i])
    emb_oof[va_i] = r.predict(E_tr[va_i]); emb_te += r.predict(E_te)/N_FOLDS
print('mpnet ridge MSE:', mean_squared_error(y, emb_oof))
P_all = PCA(n_components=32, random_state=SEED).fit_transform(E_all)
Ptr, Pte = P_all[:len(tr)], P_all[len(tr):]
'''))

C.append(nbf.v4.new_code_cell(r'''
# ---------- ozellik muhendisligi + organizator doldurma bayraklari ----------
SKILLS = ['coding_score','problem_solving_score','data_structures_score','sql_score',
          'machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
ROLE_SKILLS = {
 'Backend Developer': ['backend_score','sql_score','data_structures_score','coding_score'],
 'Frontend Developer': ['frontend_score','coding_score','problem_solving_score'],
 'Software Developer': ['coding_score','data_structures_score','problem_solving_score','backend_score','frontend_score'],
 'Data Scientist': ['machine_learning_score','sql_score','problem_solving_score'],
 'Data Analyst': ['sql_score','machine_learning_score','problem_solving_score'],
 'AI Engineer': ['machine_learning_score','coding_score','data_structures_score'],
 'Cloud Engineer': ['cloud_score','devops_score','backend_score'],
 'DevOps Engineer': ['devops_score','cloud_score','backend_score'],
 'Machine Learning Engineer': ['machine_learning_score','coding_score','data_structures_score'],
 'Full Stack Developer': ['backend_score','frontend_score','coding_score','sql_score'],
 'Mobile Developer': ['coding_score','frontend_score','problem_solving_score']}

def engineer(df):
    df = df.copy()
    df['skill_mean']=df[SKILLS].mean(axis=1); df['skill_max']=df[SKILLS].max(axis=1)
    df['skill_min']=df[SKILLS].min(axis=1); df['skill_std']=df[SKILLS].std(axis=1)
    rm=np.zeros(len(df)); rx=np.zeros(len(df))
    for role,cols in ROLE_SKILLS.items():
        m=(df['target_role']==role).values
        if m.sum(): rm[m]=df.loc[m,cols].mean(axis=1); rx[m]=df.loc[m,cols].max(axis=1)
    df['role_skill_mean']=rm; df['role_skill_max']=rx
    df['role_skill_gap']=df['role_skill_mean']-df['skill_mean']
    df['interview_mean']=df[['technical_interview_score','hr_interview_score']].mean(axis=1)
    df['soft_mean']=df[['communication_score','teamwork_score','leadership_score','presentation_score']].mean(axis=1)
    df['total_projects']=df['real_client_project_count']+df['freelance_project_count']
    df['exp_score']=(df['real_client_project_count']*2+df['freelance_project_count']
                     +df['internship_count']+df['hackathon_awards']*2)
    df['internship_total']=df['internship_count']*df['internship_duration_months'].fillna(0)
    df['github_activity']=df['github_repo_count']*(1+df['github_avg_stars'].fillna(0))
    df['interview_ratio']=df['interviews_attended']/(df['applications_sent']+1)
    df['years_since_grad']=df['application_year']-df['graduation_year']
    df['pq_x_ti']=df['project_quality_score']*df['technical_interview_score']
    df['pq_x_skill']=df['project_quality_score']*df['skill_mean']
    df['pq_x_role']=df['project_quality_score']*df['role_skill_mean']
    df['ti_x_skill']=df['technical_interview_score']*df['skill_mean']
    df['pq_x_comm']=df['project_quality_score']*df['communication_score']
    df['portfolio_x_github']=df['portfolio_score'].fillna(0)*np.log1p(df['github_repo_count'])
    df['n_missing']=df[['english_exam_score','internship_duration_months','portfolio_score',
                        'github_avg_stars','open_source_contribution_count',
                        'linkedin_profile_score','hr_interview_score']].isna().sum(axis=1)
    df['text_len']=df['mentor_feedback_text'].str.len()
    df['text_words']=df['mentor_feedback_text'].str.split().str.len()
    return df

# doldurma bayraklari: bazi degerler 2 ondalik yerine 10+ ondalikli -> organizator imputasyonu
IMP_COLS=['sql_score','machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']
def impflags(path):
    d = pd.read_csv(path, encoding='utf-8-sig', dtype=str)
    def dl(s):
        s=str(s); return len(s.split('.')[1]) if '.' in s and s!='nan' else 0
    F = pd.DataFrame({f'imp_{c}': (d[c].map(dl)>3).astype(int) for c in IMP_COLS})
    F['imp_count']=F.sum(axis=1); return F
imp_tr = impflags(f'{DATA}/train.csv'); imp_te = impflags(te_path)

CAT_COLS=['department','university_tier','target_role','hobby','preferred_social_media_platform']
DROP=['student_id','career_success_score','mentor_feedback_text']
def build(df,S_,P_,impf,tf,em,bt):
    f = engineer(df).drop(columns=[c for c in DROP if c in df.columns]).reset_index(drop=True)
    f = pd.concat([f, pd.DataFrame(S_, columns=[f'svd_{i}' for i in range(S_.shape[1])]),
                   pd.DataFrame(P_, columns=[f'emb_{i}' for i in range(P_.shape[1])]),
                   impf.reset_index(drop=True)], axis=1)
    f['text_pred_tfidf']=tf; f['text_pred_emb']=em; f['text_pred_bert']=bt
    return f
X = build(tr,Str,Ptr,imp_tr,tfidf_oof,emb_oof,bert_oof)
Xte = build(te,Ste,Pte,imp_te,tfidf_te,emb_te,bert_te)
print('feature matrix:', X.shape)
'''))

C.append(nbf.v4.new_code_cell(r'''
# ---------- GBDT toplulugu ----------
def run_cat(seed):
    oof=np.zeros(len(X)); tep=np.zeros(len(Xte))
    Xc,Xt = X.copy(),Xte.copy()
    for c in CAT_COLS: Xc[c]=Xc[c].astype(str); Xt[c]=Xt[c].astype(str)
    for tr_i,va_i in folds:
        m=CatBoostRegressor(iterations=10000,learning_rate=0.03,depth=6,l2_leaf_reg=3,
                            cat_features=CAT_COLS,early_stopping_rounds=400,random_seed=seed,verbose=False)
        m.fit(Xc.iloc[tr_i],y[tr_i],eval_set=(Xc.iloc[va_i],y[va_i]))
        oof[va_i]=m.predict(Xc.iloc[va_i]); tep+=m.predict(Xt)/N_FOLDS
    return oof,tep
def run_lgb(seed):
    oof=np.zeros(len(X)); tep=np.zeros(len(Xte))
    Xc,Xt = X.copy(),Xte.copy()
    for c in CAT_COLS:
        Xc[c]=Xc[c].astype('category'); Xt[c]=Xt[c].astype('category').cat.set_categories(Xc[c].cat.categories)
    for tr_i,va_i in folds:
        m=lgb.LGBMRegressor(n_estimators=6000,learning_rate=0.02,num_leaves=63,colsample_bytree=0.7,
                            subsample=0.8,subsample_freq=1,min_child_samples=20,reg_alpha=0.1,
                            reg_lambda=1.0,random_state=seed,verbose=-1)
        m.fit(Xc.iloc[tr_i],y[tr_i],eval_set=[(Xc.iloc[va_i],y[va_i])],
              callbacks=[lgb.early_stopping(300,verbose=False)])
        oof[va_i]=m.predict(Xc.iloc[va_i]); tep+=m.predict(Xt)/N_FOLDS
    return oof,tep
def run_xgb(seed):
    oof=np.zeros(len(X)); tep=np.zeros(len(Xte))
    Xc,Xt = X.copy(),Xte.copy()
    for c in CAT_COLS:
        Xc[c]=Xc[c].astype('category'); Xt[c]=Xt[c].astype('category').cat.set_categories(Xc[c].cat.categories)
    for tr_i,va_i in folds:
        m=xgb.XGBRegressor(n_estimators=6000,learning_rate=0.02,max_depth=6,colsample_bytree=0.7,
                           subsample=0.8,min_child_weight=5,reg_alpha=0.1,reg_lambda=1.0,
                           enable_categorical=True,tree_method='hist',early_stopping_rounds=300,
                           random_state=seed, device='cuda' if torch.cuda.is_available() else 'cpu')
        m.fit(Xc.iloc[tr_i],y[tr_i],eval_set=[(Xc.iloc[va_i],y[va_i])],verbose=False)
        oof[va_i]=m.predict(Xc.iloc[va_i]); tep+=m.predict(Xt)/N_FOLDS
    return oof,tep

oofs,teps,names=[],[],[]
for seed in (42,2026):
    for nm,fn in (('cat',run_cat),('lgb',run_lgb),('xgb',run_xgb)):
        o,t=fn(seed); oofs.append(o); teps.append(t); names.append(f'{nm}_{seed}')
        print(f'{nm}_{seed}: uniform={mean_squared_error(y,np.clip(o,0,100)):.3f} LB_est={lb_est(np.clip(o,0,100)):.3f}', flush=True)
oofs += [tfidf_oof, emb_oof, bert_oof]; teps += [tfidf_te, emb_te, bert_te]
names += ['tfidf','emb','bert']
O=np.vstack(oofs).T; T=np.vstack(teps).T
np.save('O.npy',O); np.save('T.npy',T)
'''))

C.append(nbf.v4.new_code_cell(r'''
# ---------- harman + submission + TANI RAPORU ----------
def opt(M,yt,w_eval,seeds=6):
    def loss(w):
        w=np.abs(w); w/=w.sum()
        return np.average((yt-np.clip(M@w,0,100))**2, weights=w_eval)
    best=None
    for s in range(seeds):
        r=minimize(loss,np.random.RandomState(s).rand(M.shape[1])+0.5,method='Nelder-Mead',
                   options={'maxiter':12000,'fatol':1e-9})
        if best is None or r.fun<best.fun: best=r
    w=np.abs(best.x); w/=w.sum(); return w,best.fun

yr = tr.application_year.values
w_g,f_g = opt(O,y,iw)
late = yr>=2024
w_e,_ = opt(O[~late],y[~late],None); w_l,_ = opt(O[late],y[late],None)
pred_era_oof = np.where(late, np.clip(O@w_l,0,100), np.clip(O@w_e,0,100))
f_era = lb_est(pred_era_oof)

te_late = (te.application_year>=2024).values
if f_era < f_g:
    pred_te = np.where(te_late, np.clip(T@w_l,0,100), np.clip(T@w_e,0,100)); kind='per-era'
else:
    pred_te = np.clip(T@w_g,0,100); kind='global'
sub = pd.DataFrame({'student_id': te['student_id'], 'career_success_score': pred_te})
sub.to_csv('submission.csv', index=False)

print('='*60)
print('TANI RAPORU (bunu geri gonderin)')
print('='*60)
print('BERT OOF MSE / corr: %.2f / %.4f' % (mean_squared_error(y,bert_oof), np.corrcoef(bert_oof,y)[0,1]))
print('TFIDF MSE: %.2f | mpnet MSE: %.2f' % (mean_squared_error(y,tfidf_oof), mean_squared_error(y,emb_oof)))
for nm,o in zip(names[:-3], oofs[:-3]):
    print('%s: LB_est=%.3f' % (nm, lb_est(np.clip(o,0,100))))
print('GLOBAL blend LB_est: %.3f' % f_g)
print('PER-ERA blend LB_est: %.3f' % f_era)
print('Secilen: %s | submission.csv yazildi (%d satir)' % (kind, len(sub)))
print(sub['career_success_score'].describe().round(2))
'''))

nb['cells'] = C
nbf.write(nb, 'kaggle_gpu_pipeline.ipynb')
print('notebook written')
