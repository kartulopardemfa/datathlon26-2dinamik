import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell("""# Datathon 2026 — Gemini ile Mentor Metni Okuma

**Fikir:** BERT kalıp eşleştirir; Gemini metni ANLAR. Her mentor yorumunu Gemini'ye okutup
"bu yoruma göre öğrencinin kariyer skoru kaç olur?" + güçlü/zayıf yön listesi çıkartıyoruz.

**Kurulum:**
1. https://aistudio.google.com/apikey → ücretsiz API anahtarı alın
2. Aşağıdaki hücrede `API_KEY = "..."` kısmına yapıştırın
3. Drive'da `MyDrive/datathon/` içinde `train.csv` + `test.csv` olmalı (önceki gibi)
4. Tümünü çalıştır — **GPU gerekmez**, ~1.5-2 saat (API hız limitine bağlı)

**Kopma olursa:** yeniden Tümünü çalıştır — bitmiş partiler Drive'dan okunur, devam eder.

Sonunda `llm_feats_train.csv` + `llm_feats_test.csv` üretilir (Drive'a da kaydedilir) →
bu iki dosyayı bana yükleyin."""))

C.append(nbf.v4.new_code_cell(r'''
API_KEY = "BURAYA_API_ANAHTARINIZI_YAPISTIRIN"

import subprocess, sys
try:
    from google import genai
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','google-genai'])
    from google import genai
import glob, os, json, time, re
import numpy as np, pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print('drive mount atlandi:', e)
CKPT='/content/drive/MyDrive/datathon_gemini'
os.makedirs(CKPT, exist_ok=True)

paths = (glob.glob('/content/train.csv') + glob.glob('/content/drive/MyDrive/**/train.csv', recursive=True))
assert paths, 'train.csv bulunamadi'
DATA = os.path.dirname(paths[0])
tr = pd.read_csv(f'{DATA}/train.csv', encoding='utf-8-sig')
te_path = f'{DATA}/test.csv' if os.path.exists(f'{DATA}/test.csv') else glob.glob(f'{DATA}/test*.csv')[0]
te = pd.read_csv(te_path, encoding='utf-8-sig')
print(tr.shape, te.shape)

client = genai.Client(api_key=API_KEY)
MODEL_ID = 'gemini-2.5-flash'

ASPECTS = ['kodlama','problem_cozme','veri_yapilari','sql','makine_ogrenmesi','backend',
           'frontend','bulut','devops','proje_kalitesi','staj','github_acik_kaynak',
           'portfolyo','iletisim','takim_calismasi','liderlik','sunum','mulakat']

PROMPT = """Asagida bir is/kariyer mentorunun ogrenciler hakkinda yazdigi degerlendirme yorumlari var.
Her yorum icin SADECE yorumun diline/tonuna/icerigine bakarak su alanlari doldur:
- "puan": yorumun ima ettigi kariyer basari skoru (0-100 arasi tam sayi; cok ovgu dolu=85-100, dengeli=55-80, agirlikli elestirel=20-55)
- "ton": -2 (cok olumsuz) ile +2 (cok olumlu) arasi tam sayi
- "guclu": su listeden yorumda GUCLU olarak gecen yonler: {aspects}
- "zayif": ayni listeden GELISTIRILMESI gereken yonler
- "kesinlik": mentorun ifadesindeki kesinlik/guven 1-5 (1=cok temkinli, 5=cok emin)

Cikti formati: JSON dizisi, her eleman {{"id": <verilen id>, "puan": int, "ton": int, "guclu": [...], "zayif": [...], "kesinlik": int}}
SADECE JSON ver, baska aciklama yazma.

Yorumlar:
{items}"""

def annotate_batch(ids, texts, retries=4):
    items = '\n'.join(f'[id={i}] {t}' for i, t in zip(ids, texts))
    prompt = PROMPT.format(aspects=', '.join(ASPECTS), items=items)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL_ID, contents=prompt,
                config={'temperature': 0, 'max_output_tokens': 8000})
            txt = resp.text
            m = re.search(r'\[.*\]', txt, re.S)
            arr = json.loads(m.group(0))
            out = {int(o['id']): o for o in arr if 'id' in o}
            if len(out) >= len(ids) - 2:
                return out
        except Exception as e:
            msg = str(e)
            if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
                time.sleep(20 + 10*attempt)
            else:
                time.sleep(3)
    return {}

def run_split(name, texts):
    BATCH = 20
    n = len(texts)
    nb_ = (n + BATCH - 1)//BATCH
    results = {}
    for b in range(nb_):
        ck = f'{CKPT}/{name}_b{b}.json'
        if os.path.exists(ck):
            with open(ck) as f: results.update({int(k):v for k,v in json.load(f).items()})
            continue
        lo, hi = b*BATCH, min((b+1)*BATCH, n)
        out = annotate_batch(list(range(lo,hi)), [str(t)[:600] for t in texts[lo:hi]])
        with open(ck,'w') as f: json.dump(out, f)
        results.update(out)
        if b % 25 == 0:
            print(f'{name}: {b}/{nb_} parti, ornek:', list(out.values())[:1], flush=True)
        time.sleep(1.0)
    return results

def to_feats(results, n):
    rows=[]
    for i in range(n):
        o = results.get(i, {})
        r = {'llm_puan': o.get('puan', np.nan), 'llm_ton': o.get('ton', np.nan),
             'llm_kesinlik': o.get('kesinlik', np.nan),
             'llm_n_guclu': len(o.get('guclu',[])), 'llm_n_zayif': len(o.get('zayif',[]))}
        for a in ASPECTS:
            r[f'llm_g_{a}'] = int(a in o.get('guclu',[]))
            r[f'llm_z_{a}'] = int(a in o.get('zayif',[]))
        rows.append(r)
    return pd.DataFrame(rows)
'''))

C.append(nbf.v4.new_code_cell(r'''
res_tr = run_split('train', tr['mentor_feedback_text'].fillna('').values)
F_tr = to_feats(res_tr, len(tr))
F_tr.to_csv('llm_feats_train.csv', index=False); F_tr.to_csv(f'{CKPT}/llm_feats_train.csv', index=False)
y = tr['career_success_score'].values
ok = F_tr['llm_puan'].notna()
print('kapsam: %.1f%%' % (100*ok.mean()))
print('corr(llm_puan, gercek skor): %.4f' % np.corrcoef(F_tr.loc[ok,'llm_puan'], y[ok.values])[0,1])
print('llm_puan MSE: %.2f' % ((F_tr.loc[ok,'llm_puan']-y[ok.values])**2).mean())
'''))

C.append(nbf.v4.new_code_cell(r'''
res_te = run_split('test', te['mentor_feedback_text'].fillna('').values)
F_te = to_feats(res_te, len(te))
F_te.to_csv('llm_feats_test.csv', index=False); F_te.to_csv(f'{CKPT}/llm_feats_test.csv', index=False)
print('='*50)
print('GEMINI RAPORU (bana gonderin):')
ok = F_tr['llm_puan'].notna()
print('train kapsam %.1f%% | corr(llm_puan,y)=%.4f' % (100*ok.mean(),
      np.corrcoef(F_tr.loc[ok,'llm_puan'], y[ok.values])[0,1]))
print('dosyalar: llm_feats_train.csv, llm_feats_test.csv (Drive: datathon_gemini/)')
'''))
nb['cells']=C
nbf.write(nb,'colab_gemini.ipynb')
print('written')
