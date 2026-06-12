import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell("""# Datathon 2026 — BÜYÜK FÜZYON (Colab A100)

**Kurulum (Colab):**
1. Çalışma zamanı → Çalışma zamanı türünü değiştir → **A100 GPU** (yoksa L4)
2. Sol panel → Dosyalar → `train.csv` ve `test.csv` dosyalarını sürükleyip `/content`e yükleyin
3. Çalışma zamanı → **Tümünü çalıştır** (~2-2.5 saat A100'de)

Üretilenler: `bigfusion_oof.npy`, `bigfusion_te.npy`, `submission_bigfusion.csv` + rapor.
Bittiğinde rapor bloğunu bana gönderin, npy dosyalarını indirip yükleyin."""))

C.append(nbf.v4.new_code_cell(r'''
import subprocess, sys
try:
    import transformers, sentence_transformers
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers','sentence-transformers','catboost'])
import glob, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

paths = (glob.glob('/kaggle/input/*/train.csv') + glob.glob('/kaggle/input/*/*/train.csv')
         + glob.glob('/content/train.csv') + glob.glob('/content/drive/MyDrive/**/train.csv', recursive=True))
assert paths, 'train.csv bulunamadi: /content e yukleyin'
DATA = os.path.dirname(paths[0])
tr = pd.read_csv(f'{DATA}/train.csv', encoding='utf-8-sig')
te_path = f'{DATA}/test.csv' if os.path.exists(f'{DATA}/test.csv') else glob.glob(f'{DATA}/test*.csv')[0]
te = pd.read_csv(te_path, encoding='utf-8-sig')
y = tr['career_success_score'].values.astype(np.float64)
SEED=42; N_FOLDS=5
folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(tr))
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_/trp).values
def lb_est(p): return np.average((y-p)**2, weights=iw)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
print(tr.shape, te.shape, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK')
'''))

C.append(nbf.v4.new_code_cell(r'''
# BUYUK FUZYON: xlm-roberta-large + cross-attention tablo birlesimi
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

MODEL='xlm-roberta-large'; MAXLEN=160; BS=16; EPOCHS=3; ACCUM=2
tok = AutoTokenizer.from_pretrained(MODEL)
Y_MU, Y_SD = float(y.mean()), float(y.std())
tx_tr = tr['mentor_feedback_text'].fillna('').values
tx_te = te['mentor_feedback_text'].fillna('').values
yz = (y-Y_MU)/Y_SD

CAT_COLS=['department','university_tier','target_role','hobby','preferred_social_media_platform']
num_cols=[c for c in tr.columns if c not in CAT_COLS+['student_id','career_success_score','mentor_feedback_text']]
imp_ = SimpleImputer(strategy='median').fit(tr[num_cols])
ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(tr[CAT_COLS])
sc_ = StandardScaler().fit(np.hstack([imp_.transform(tr[num_cols]), ohe_.transform(tr[CAT_COLS])]))
TAB_tr = sc_.transform(np.hstack([imp_.transform(tr[num_cols]), ohe_.transform(tr[CAT_COLS])])).astype(np.float32)
TAB_te = sc_.transform(np.hstack([imp_.transform(te[num_cols]), ohe_.transform(te[CAT_COLS])])).astype(np.float32)
TD = TAB_tr.shape[1]

class DS(Dataset):
    def __init__(self, texts, tab, targets=None):
        self.enc = tok(list(texts), truncation=True, max_length=MAXLEN, padding='max_length', return_tensors='np')
        self.tab = tab; self.t = targets
    def __len__(self): return len(self.tab)
    def __getitem__(self, i):
        d = {k: torch.tensor(v[i]) for k,v in self.enc.items() if k in ('input_ids','attention_mask')}
        d['tab'] = torch.tensor(self.tab[i])
        if self.t is not None: d['target'] = torch.tensor(np.float32(self.t[i]))
        return d

class XFusion(nn.Module):
    """Tablo vektoru sorgu (query) olur, metin tokenlarina cross-attention ile bakar."""
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL)
        h = self.bb.config.hidden_size
        self.tabq = nn.Sequential(nn.Linear(TD,512), nn.GELU(), nn.Linear(512,h))
        self.xattn = nn.MultiheadAttention(h, num_heads=8, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2*h+h//2*0+h), nn.Linear(2*h,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, input_ids, attention_mask, tab):
        o = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (o*m).sum(1)/m.sum(1)
        q = self.tabq(tab).unsqueeze(1)
        att,_ = self.xattn(q, o, o, key_padding_mask=(attention_mask==0))
        z = torch.cat([pooled, att.squeeze(1)], dim=1)
        return self.head[1:](z).squeeze(-1) if False else nn.functional.linear(z, torch.empty(0)) if False else self._h(z)
    def _h(self, z):
        return self.head_seq(z)

# basit ve saglam: head'i ayri tanimla
class XFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL)
        h = self.bb.config.hidden_size
        self.tabq = nn.Sequential(nn.Linear(TD,512), nn.GELU(), nn.Linear(512,h))
        self.xattn = nn.MultiheadAttention(h, num_heads=8, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2*h,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, input_ids, attention_mask, tab):
        o = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (o*m).sum(1)/m.sum(1)
        q = self.tabq(tab).unsqueeze(1)
        att,_ = self.xattn(q, o, o, key_padding_mask=(attention_mask==0))
        return self.head(torch.cat([pooled, att.squeeze(1)], dim=1)).squeeze(-1)

def train_seed(seed):
    torch.manual_seed(seed)
    te_dl = DataLoader(DS(tx_te, TAB_te), batch_size=64)
    f_oof = np.zeros(len(tr)); f_te = np.zeros(len(te))
    scaler = torch.cuda.amp.GradScaler()
    for fold,(tr_i,va_i) in enumerate(folds):
        model = XFusion().to(DEV)
        opt_ = torch.optim.AdamW([{'params': model.bb.parameters(),'lr':1e-5},
                                  {'params': list(model.tabq.parameters())+list(model.xattn.parameters())+list(model.head.parameters()),'lr':5e-4}])
        dl = DataLoader(DS(tx_tr[tr_i], TAB_tr[tr_i], yz[tr_i]), batch_size=BS, shuffle=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_, T_max=EPOCHS*len(dl)//ACCUM)
        lossf = nn.MSELoss(); model.train()
        for ep in range(EPOCHS):
            opt_.zero_grad()
            for step,b in enumerate(dl):
                with torch.cuda.amp.autocast():
                    p = model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV))
                    loss = lossf(p, b['target'].to(DEV))/ACCUM
                scaler.scale(loss).backward()
                if (step+1)%ACCUM==0:
                    scaler.step(opt_); scaler.update(); opt_.zero_grad(); sched.step()
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            va_dl = DataLoader(DS(tx_tr[va_i], TAB_tr[va_i]), batch_size=64)
            pv = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV)).float().cpu().numpy() for b in va_dl])
            f_oof[va_i] = pv*Y_SD+Y_MU
            pt = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV)).float().cpu().numpy() for b in te_dl])
            f_te += (pt*Y_SD+Y_MU)/N_FOLDS
        print(f'seed {seed} fold {fold}: val MSE {mean_squared_error(y[va_i], f_oof[va_i]):.2f}', flush=True)
        del model; torch.cuda.empty_cache()
    return f_oof, f_te

R = {s: train_seed(s) for s in (42, 7)}
bf_oof = np.mean([R[s][0] for s in R], axis=0)
bf_te = np.mean([R[s][1] for s in R], axis=0)
np.save('bigfusion_oof.npy', bf_oof); np.save('bigfusion_te.npy', bf_te)
oc = np.clip(bf_oof,0,100)
print('='*60)
print('BUYUK FUZYON RAPORU (bana gonderin)')
print('uniform MSE=%.3f  LB_est=%.3f  corr=%.4f' % (mean_squared_error(y,oc), lb_est(oc), np.corrcoef(bf_oof,y)[0,1]))
pd.DataFrame({'student_id':te['student_id'],'career_success_score':np.clip(bf_te,0,100)}).to_csv('submission_bigfusion.csv', index=False)
print('dosyalar: bigfusion_oof.npy, bigfusion_te.npy, submission_bigfusion.csv')
'''))
nb['cells']=C
nbf.write(nb, 'colab_bigfusion.ipynb')
print('written')
