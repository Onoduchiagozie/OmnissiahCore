"""Live API retrieval + real cached reranker + staged selection. No generation.

Does not instantiate a second embedding model or FAISS/BM25 index and cannot
rebuild them. Full passages stay in Oracle staging; summary omits corpus text.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ['OMNISSIAH_ACTIVE_PROFILE'] = 'dell_query'
os.environ['HF_HOME'] = '/opt/omnissiah/models/huggingface'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

APP = Path('/opt/omnissiah/app')
STAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import requests
import torch
from sentence_transformers import CrossEncoder
from Core.config_loader import paths, retrieval_cfg
from Core.retriever import OmnissiahRetriever as Deployed


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def compact(hit):
    return {key: hit.get(key) for key in ('chunk_id', 'source', 'stitch_range', 'rerank_score', 'rerank_scope', 'pair_tokens')}


def main():
    torch.set_num_threads(2)
    staged = load('staged_raw', STAGE/'retriever.py')
    rewriter = load('scene_query_plan', STAGE.parent/'natural-queries/query_rewriter.py').QueryRewriter()
    cases = json.loads((STAGE/'scene_cases.json').read_text())
    cases += [
        {'id': 'negative-wifi', 'raw_query': 'how do I reset my wifi router?'},
        {'id': 'laer-blade', 'raw_query': 'what is the laer blade?'},
        {'id': 'comparison', 'raw_query': 'ferrus vs fulgrim'},
    ]
    pool = retrieval_cfg['candidate_pool']
    top_k = retrieval_cfg['top_k']
    window = retrieval_cfg['stitching_window']
    raw_rows = []
    needed = set()
    for case in cases:
        original = case['raw_query']
        plan = rewriter.rewrite(original)
        variants = [('original', original)]
        if plan.dense_query != original:
            variants.append(('normalized', plan.dense_query))
        for label, query in variants:
            response = requests.post('http://127.0.0.1:8000/query/inspect', json={
                'query': query, 'top_k': pool, 'candidate_pool': pool,
            }, timeout=(10, 180))
            response.raise_for_status()
            candidates = response.json()['inspection']['grounded_hits']
            for hit in candidates:
                cid = hit.get('chunk_id')
                if cid is not None:
                    needed.update(range(max(0, cid-window), cid+window+1))
            raw_rows.append({'case': case['id'], 'variant': label, 'query': query, 'original_query': original, 'candidates': candidates})
            (STAGE/'retrieval-candidates.json').write_text(json.dumps(raw_rows), encoding='utf-8')
            print(json.dumps({'retrieved': case['id'], 'variant': label, 'candidates': len(candidates)}), flush=True)

    # Retain only the neighbors required for these bounded candidates.
    with open(paths['metadata'], encoding='utf-8') as handle:
        all_metadata = json.load(handle)
    metadata = [dict(m, chunk_id=m.get('chunk_id', i)) for i, m in enumerate(all_metadata) if m.get('chunk_id', i) in needed]
    del all_metadata
    reranker = CrossEncoder(retrieval_cfg['rerank_model'], device='cpu')
    old = Deployed.__new__(Deployed)
    new = staged.OmnissiahRetriever.__new__(staged.OmnissiahRetriever)
    for obj in (old, new):
        obj.metadata = metadata
        obj._id_to_idx = {m['chunk_id']: i for i, m in enumerate(metadata)}
        obj.reranker = reranker
    results = []
    for row in raw_rows:
        started = time.monotonic()
        query = row['query']
        candidates = row.pop('candidates')
        stitched = old._stitch_chunks(copy.deepcopy(candidates), window)
        if len(stitched) > 1:
            baseline = [h for h in old._rerank(query, stitched, top_k) if h['rerank_score'] >= 0]
        else:
            baseline = stitched[:top_k]
        reranked, selected = new._select_candidates(query, copy.deepcopy(candidates), top_k, window)
        for hit in reranked:
            hit['pair_tokens'] = len(reranker.tokenizer(query, hit['text'], truncation=False)['input_ids'])
        row.update(candidate_count=len(candidates), deployed_selection=baseline,
                   staged_selection=selected, raw_ranking=reranked,
                   seconds=round(time.monotonic()-started, 2))
        results.append(row)
        print(json.dumps({'case': row['case'], 'variant': row['variant'],
                          'deployed': [compact(h) for h in baseline],
                          'staged': [compact(h) for h in selected], 'seconds': row['seconds']}), flush=True)
        (STAGE/'live-selection-full.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

    summary = {
        'scope': 'Live deployed FAISS/BM25 retrieval candidates; real offline CrossEncoder; staged selection/stitching. No staged API or generation accuracy proof.',
        'model': retrieval_cfg['rerank_model'], 'max_length': reranker.max_length,
        'candidate_pool': pool, 'top_k': top_k, 'stitching_window': window,
        'staged_sha256': hashlib.sha256((STAGE/'retriever.py').read_bytes()).hexdigest(),
        'deployed_sha256': hashlib.sha256((APP/'Core/retriever.py').read_bytes()).hexdigest(),
        'rows': [{**{k: r[k] for k in ('case', 'variant', 'query', 'original_query', 'candidate_count', 'seconds')},
                  'deployed': [compact(h) for h in r['deployed_selection']],
                  'staged': [compact(h) for h in r['staged_selection']],
                  'raw_top': [compact(h) for h in r['raw_ranking'][:8]],
                  'source_verified_answer': False} for r in results],
    }
    (STAGE/'live-selection-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Completed live selection comparison; no LLM calls.', flush=True)


if __name__ == '__main__':
    main()
