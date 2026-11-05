"""Compare query forms on a bounded sample of the live corpus, no generation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import requests

os.environ['HF_HOME'] = '/opt/omnissiah/models/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
from sentence_transformers import CrossEncoder

stage = Path('/opt/omnissiah/staging/natural-queries')
spec = importlib.util.spec_from_file_location('query_rewriter', stage/'query_rewriter.py')
module = importlib.util.module_from_spec(spec)
sys.modules['query_rewriter'] = module
spec.loader.exec_module(module)
rewriter = module.QueryRewriter()
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
queries = [
    'ferrus vs fulgrim',
    'what happened on istvan five?',
    'why did horus turn traitor?',
    'what is the laer blade?',
    'tell me about the emperor and magnus',
    'first shot fired of the dropsite assault on istvan five',
    'how do I reset my wifi router?',
]
rows = []
for original in queries:
    plan = rewriter.rewrite(original)
    variants = [('original', original)]
    if plan.dense_query != original:
        variants.append(('rewritten', plan.dense_query))
    comparisons = []
    for label, query in variants:
        started = time.monotonic()
        response = requests.post('http://127.0.0.1:8000/query/inspect', json={
            'query':query, 'candidate_pool':80, 'top_k':20,
        }, timeout=120)
        response.raise_for_status()
        candidates = response.json()['inspection']['stitched_hits']
        if candidates:
            scores = reranker.predict([[query, c['text']] for c in candidates])
            ranked = sorted(zip(scores, candidates), key=lambda pair:float(pair[0]), reverse=True)
        else:
            ranked = []
        comparisons.append({
            'variant':label, 'retrieval_query':query,
            'sample_size':len(candidates),
            'nonnegative_scores':sum(float(score)>=0 for score,_ in ranked),
            'top':[{'score':round(float(score),3), 'source':c['source'],
                    'preview':c['text'][:260].replace('\n',' ')} for score,c in ranked[:2]],
            'seconds':round(time.monotonic()-started,2),
        })
    row = {'original_query':original, 'plan':plan.to_dict(), 'comparisons':comparisons}
    rows.append(row)
    print(json.dumps(row),flush=True)
(stage/'corpus-comparison.json').write_text(json.dumps(rows,indent=2))
print('Completed corpus comparison. This samples live inspection candidates; it is not a full staged production or answer-accuracy test.',flush=True)
