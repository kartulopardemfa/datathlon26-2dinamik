"""Fine-tune multilingual MiniLM for regression on mentor_feedback_text (CPU)."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAXLEN = 96
EPOCHS = 2
BS = 32
LR = 3e-5

torch.manual_seed(42)
torch.set_num_threads(4)

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
y_raw = tr['career_success_score'].values.astype(np.float32)
Y_MU, Y_SD = float(y_raw.mean()), float(y_raw.std())
y = (y_raw - Y_MU) / Y_SD
tok = AutoTokenizer.from_pretrained(MODEL)

class DS(Dataset):
    def __init__(self, texts, targets=None):
        self.enc = tok(list(texts), truncation=True, max_length=MAXLEN,
                       padding='max_length', return_tensors='np')
        self.targets = targets
    def __len__(self):
        return len(self.enc['input_ids'])
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        if self.targets is not None:
            item['target'] = torch.tensor(self.targets[i])
        return item

class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bb = AutoModel.from_pretrained(MODEL)
        # freeze token embeddings (250k vocab dominates params; big CPU speedup)
        self.bb.embeddings.word_embeddings.weight.requires_grad = False
        self.head = nn.Linear(self.bb.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask, **kw):
        out = self.bb(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1)
        return self.head(pooled).squeeze(-1)

texts_tr = tr['mentor_feedback_text'].fillna('').values
texts_te = te['mentor_feedback_text'].fillna('').values
te_ds = DS(texts_te)
te_dl = DataLoader(te_ds, batch_size=64)

kf = KFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(tr)); tep = np.zeros(len(te))
for fold, (tr_i, va_i) in enumerate(kf.split(texts_tr)):
    model = Reg()
    opt = torch.optim.AdamW([
        {'params': model.bb.parameters(), 'lr': LR},
        {'params': model.head.parameters(), 'lr': 1e-3},
    ])
    dl = DataLoader(DS(texts_tr[tr_i], y[tr_i]), batch_size=BS, shuffle=True)
    lossf = nn.MSELoss()
    model.train()
    for ep in range(EPOCHS):
        tot = 0
        import time; t0 = time.time()
        for step, b in enumerate(dl):
            opt.zero_grad()
            p = model(b['input_ids'], b['attention_mask'])
            loss = lossf(p, b['target'])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b['target'])
            if step % 50 == 49:
                print(f'fold {fold} ep {ep} step {step+1}/{len(dl)} '
                      f'({(time.time()-t0)/(step+1):.2f}s/step)', flush=True)
        print(f'fold {fold} ep {ep} train mse {tot/len(tr_i):.2f}', flush=True)
    model.eval()
    with torch.no_grad():
        va_dl = DataLoader(DS(texts_tr[va_i]), batch_size=64)
        preds = np.concatenate([model(b['input_ids'], b['attention_mask']).numpy() for b in va_dl])
        preds = preds * Y_SD + Y_MU
        oof[va_i] = preds
        tp = np.concatenate([model(b['input_ids'], b['attention_mask']).numpy() for b in te_dl])
        tep += (tp * Y_SD + Y_MU) / 5
    print(f'fold {fold} val mse {mean_squared_error(y_raw[va_i], preds):.2f}', flush=True)
    np.save(f'cache_ft_oof_f{fold}.npy', oof)
    np.save(f'cache_ft_te_f{fold}.npy', tep)

print('FT text OOF MSE:', mean_squared_error(y_raw, oof),
      'corr:', np.corrcoef(oof, y_raw)[0, 1], flush=True)
np.save('cache_ft_oof.npy', oof)
np.save('cache_ft_te.npy', tep)
