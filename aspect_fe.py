import re
import numpy as np
import pandas as pd

ASPECTS = {
 'kodlama': ['kodlama', 'coding'],
 'problem': ['problem çözme', 'problem-çözme'],
 'veri_yapi': ['veri yapıları'],
 'sql': ['sql'],
 'ml': ['makine öğren', 'veri bilimi', 'yapay zeka'],
 'backend': ['backend'],
 'frontend': ['frontend'],
 'cloud': ['bulut', 'cloud'],
 'devops': ['devops'],
 'proje': ['proje kalite', 'projesinin kalite', 'proje bazında', 'müşteri proje'],
 'staj': ['staj'],
 'github': ['github', 'açık kaynak'],
 'portfoy': ['portföy', 'portfolyo'],
 'iletisim': ['iletişim'],
 'takim': ['takım', 'ekip', 'işbirliği'],
 'liderlik': ['liderlik'],
 'sunum': ['sunum'],
 'mulakat': ['mülakat', 'görüşme'],
 'ingilizce': ['ingilizce'],
 'cv': ['özgeçmiş', 'cv kalite', 'linkedin'],
}

POS = ['dikkat çek', 'başarı', 'yetkin', 'mükemmel', 'güçlü', 'öne çık', 'kayda değer',
       'olumlu', 'etkileyici', 'değer kat', 'umut verici', 'üst düzey', 'ileri düzey',
       'sevindirici', 'avantaj', 'potansiyel', 'uzman', 'iyi', 'tutku', 'sağlam',
       'parlak', 'takdir', 'istikrarlı', 'donanımlı']
NEG = ['geliştirmesi gerek', 'çalışması gerek', 'çalışması faydalı', 'eksik', 'azlığı',
       'artırmak', 'geliştirmek', 'odaklanmak', 'odaklanması', 'önerilir', 'zayıf',
       'yetersiz', 'gerekiyor', 'gerekecek', 'ihtiyaç', 'daha fazla', 'sınırlı',
       'iyileştir', 'güçlendirme', 'kaydetmesi']

def sentence_sentiment(s):
    p = sum(1 for w in POS if w in s)
    n = sum(1 for w in NEG if w in s)
    if p > n: return 1
    if n > p: return -1
    return 0

def aspect_features(texts):
    rows = []
    for t in texts.fillna('').str.lower():
        # split into clauses on sentence ends and contrast markers
        parts = re.split(r'(?<=[.!?])\s+|ancak,?|bununla birlikte,?|fakat,?|;', t)
        feats = {f'asp_{a}': 0.0 for a in ASPECTS}
        pos_cnt = neg_cnt = 0
        for s in parts:
            if not s.strip(): continue
            sent = sentence_sentiment(s)
            if sent > 0: pos_cnt += 1
            elif sent < 0: neg_cnt += 1
            for a, kws in ASPECTS.items():
                if any(k in s for k in kws):
                    feats[f'asp_{a}'] += sent
        feats['sent_pos'] = pos_cnt
        feats['sent_neg'] = neg_cnt
        feats['sent_net'] = pos_cnt - neg_cnt
        rows.append(feats)
    return pd.DataFrame(rows)
