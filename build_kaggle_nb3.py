import nbformat as nbf, re
# reuse v2 cells, replace BERT cell with FUSION cell and adjust blend/report
src = open('build_kaggle_nb2.py').read()
m = re.findall(r"C\.append\(nbf\.v4\.new_code_cell\(r'''(.*?)'''\)\)", src, flags=re.S)
setup, bert, textfeat, fe, models, blend = m

fusion = r'''
# FUZYON MODELI: BERTurk(metin) + tablo vektoru AYNI agda (GPU, ~30 dk)
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

MODEL='dbmdz/bert-base-turkish-cased'; MAXLEN=160; BS=32; EPOCHS=4
torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
Y_MU, Y_SD = float(y.mean()), float(y.std())
tx_tr = tr['mentor_feedback_text'].fillna('').values
tx_te = te['mentor_feedback_text'].fillna('').values
yz = (y-Y_MU)/Y_SD

# tablo vektoru: ham sayisal + one-hot kategorik
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

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL)
        h = self.bb.config.hidden_size
        self.tabnet = nn.Sequential(nn.Linear(TD,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(h+128,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,1))
    def forward(self, input_ids, attention_mask, tab):
        o = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (o*m).sum(1)/m.sum(1)
        return self.head(torch.cat([pooled, self.tabnet(tab)], dim=1)).squeeze(-1)

te_dl = DataLoader(DS(tx_te, TAB_te), batch_size=128)
fus_oof = np.zeros(len(tr)); fus_te = np.zeros(len(te))
for fold,(tr_i,va_i) in enumerate(folds):
    model = Fusion().to(DEV)
    opt = torch.optim.AdamW([{'params': model.bb.parameters(),'lr':2e-5},
                             {'params': list(model.tabnet.parameters())+list(model.head.parameters()),'lr':1e-3}])
    dl = DataLoader(DS(tx_tr[tr_i], TAB_tr[tr_i], yz[tr_i]), batch_size=BS, shuffle=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*len(dl))
    lossf = nn.MSELoss(); model.train()
    for ep in range(EPOCHS):
        for b in dl:
            opt.zero_grad()
            p = model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV))
            loss = lossf(p, b['target'].to(DEV)); loss.backward(); opt.step(); sched.step()
    model.eval()
    with torch.no_grad():
        va_dl = DataLoader(DS(tx_tr[va_i], TAB_tr[va_i]), batch_size=128)
        pv = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV)).cpu().numpy() for b in va_dl])
        fus_oof[va_i] = pv*Y_SD+Y_MU
        pt = np.concatenate([model(b['input_ids'].to(DEV), b['attention_mask'].to(DEV), b['tab'].to(DEV)).cpu().numpy() for b in te_dl])
        fus_te += (pt*Y_SD+Y_MU)/N_FOLDS
    print(f'FUZYON fold {fold}: val MSE {mean_squared_error(y[va_i], fus_oof[va_i]):.2f} LB_est {lb_est(np.clip(np.where(np.arange(len(y))[:,None]==None,0,fus_oof),0,100)) if False else 0:.0f}', flush=True)
    del model; torch.cuda.empty_cache()
RESULTS = globals().get('RESULTS', {})
RESULTS['fusion_mse'] = mean_squared_error(y, np.clip(fus_oof,0,100))
RESULTS['fusion_lb'] = lb_est(np.clip(fus_oof,0,100))
RESULTS['fusion_corr'] = np.corrcoef(fus_oof, y)[0,1]
print('FUZYON OOF: uniform=%.3f LB_est=%.3f corr=%.4f' % (RESULTS['fusion_mse'], RESULTS['fusion_lb'], RESULTS['fusion_corr']))
np.save('fusion_oof.npy', fus_oof); np.save('fusion_te.npy', fus_te)
bert_oof, bert_te = fus_oof, fus_te   # asagidaki hucreler bert_* adini kullaniyor
RESULTS['bert_mse']=RESULTS['fusion_mse']; RESULTS['bert_corr']=RESULTS['fusion_corr']
'''

models3 = models.replace("('catNT_42',lambda: run_cat(42,native_text=True)),\n      ", "")
models3 = models3.replace("for nm in ['cat_42','catNT_42','cat_2026','lgb_42','xgb_42']:", "for nm in ['cat_42','cat_2026','lgb_42','xgb_42']:") if "catNT_42" in models3 else models3

blend3 = blend.replace("for nm in ['cat_42','catNT_42','cat_2026','lgb_42','xgb_42']:",
                       "for nm in ['cat_42','cat_2026','lgb_42','xgb_42']:")
blend3 = blend3.replace("TANI RAPORU v2", "TANI RAPORU v3 (FUZYON)")
blend3 = blend3.replace("print('BERT: MSE=%.2f corr=%.4f' % (RESULTS['bert_mse'], RESULTS['bert_corr']))",
                        "print('FUZYON: MSE=%.2f corr=%.4f LB_est=%.3f' % (RESULTS['fusion_mse'], RESULTS['fusion_corr'], RESULTS['fusion_lb']))")

nb = nbf.v4.new_notebook()
C = [nbf.v4.new_markdown_cell("""# Datathon 2026 — v3 FÜZYON Notebook (BERTurk + Tablo, tek model)

**Mekanizma kanıtı:** corr(metin, tablo-artığı)=0.20-0.24 — metin, tabloda olmayan sinyal taşıyor
ama bağlam (rol/profil) olmadan tam çözülemiyor. Bu notebook metni ve tabloyu AYNI ağda birleştirir.

**Kurulum:** GPU + Internet ON + yarışma verisi → Run All (~1.5 saat). Sonunda `submission.csv` + TANI RAPORU v3."""),
     nbf.v4.new_code_cell(setup),
     nbf.v4.new_code_cell(fusion),
     nbf.v4.new_code_cell(textfeat),
     nbf.v4.new_code_cell(fe),
     nbf.v4.new_code_cell(models3),
     nbf.v4.new_code_cell(blend3)]
nb['cells'] = C
nbf.write(nb, 'kaggle_fusion_v3.ipynb')
print('written')
