import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
m = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
Etr = m.encode(tr['mentor_feedback_text'].fillna('').tolist(), batch_size=64, show_progress_bar=False)
np.save('cache_emb2_tr.npy', Etr)
print('train done', flush=True)
Ete = m.encode(te['mentor_feedback_text'].fillna('').tolist(), batch_size=64, show_progress_bar=False)
np.save('cache_emb2_te.npy', Ete)
print('mpnet done', Ete.shape, flush=True)
