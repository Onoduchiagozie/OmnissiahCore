"""Selection/stitching unit tests. Synthetic retrieval and scores; no accuracy claim."""
import importlib.util
from pathlib import Path
import sys
import unittest

APP = Path('/opt/omnissiah/app')
if not APP.exists():
    APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))
spec = importlib.util.spec_from_file_location('raw_rerank', Path(__file__).with_name('retriever.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectionTests(unittest.TestCase):
    def make(self, docs, scores=None):
        r = module.OmnissiahRetriever.__new__(module.OmnissiahRetriever)
        r.metadata = docs
        r._id_to_idx = {d['chunk_id']: i for i, d in enumerate(docs)}
        r.bm25 = None
        r._faiss_search = lambda query, k: [dict(d) for d in docs]
        if scores is None:
            r.reranker = None
        else:
            self.seen = []
            def predict(pairs):
                self.seen.extend(pairs)
                return [scores[text] for _, text in pairs]
            r.reranker = type('Scorer', (), {'predict': staticmethod(predict)})()
        return r

    @staticmethod
    def doc(cid, text, source='Book A'):
        return {'chunk_id': cid, 'text': text, 'source': source}

    def test_reranker_reads_original_chunks_before_neighbors(self):
        docs = [self.doc(10, 'earlier context'), self.doc(11, 'fight anchor'), self.doc(12, 'later context')]
        r = self.make(docs, {'earlier context': -1, 'fight anchor': 7, 'later context': -2})
        hits = r.search('fight', top_k=1, candidate_pool=10, stitching_window=1)
        self.assertEqual({p[1] for p in self.seen}, {d['text'] for d in docs})
        self.assertEqual(hits[0]['chunk_id'], 11)
        self.assertEqual(hits[0]['text'], 'earlier context\n\nfight anchor\n\nlater context')
        self.assertEqual(hits[0]['rerank_scope'], 'raw_chunk')

    def test_stitching_cannot_cross_book_boundary(self):
        docs = [self.doc(10, 'wrong earlier book', 'Other'), self.doc(11, 'anchor'), self.doc(12, 'same source'), self.doc(13, 'wrong later book', 'Other')]
        r = self.make(docs)
        out = r._stitch_chunks([docs[1]], window=2)
        self.assertEqual(out[0]['text'], 'anchor\n\nsame source')
        self.assertEqual(out[0]['stitch_range'], 'chunk_id 11-12')

    def test_single_candidate_is_scored_and_rejected(self):
        r = self.make([self.doc(1, 'unrelated')], {'unrelated': -1})
        self.assertEqual(r.search('fight', stitching_window=0), [])
        self.assertEqual(len(self.seen), 1)

    def test_zero_window_is_preserved(self):
        docs = [self.doc(1, 'neighbor'), self.doc(2, 'anchor')]
        r = self.make(docs, {'neighbor': -1, 'anchor': 5})
        self.assertEqual(r.search('anchor', stitching_window=0)[0]['text'], 'anchor')

    def test_overlapping_candidates_do_not_inherit_absent_anchor_score(self):
        docs = [self.doc(i, f'text {i}') for i in range(1, 6)]
        r = self.make(docs)
        out = r._stitch_chunks([docs[1], docs[2]], window=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['chunk_id'], 2)
        self.assertIn('text 2', out[0]['text'])

    def test_duplicate_groups_are_backfilled_from_remaining_ranked_candidates(self):
        docs = [self.doc(1, 'best'), self.doc(2, 'overlap'), self.doc(10, 'distinct')]
        r = self.make(docs, {'best': 9, 'overlap': 8, 'distinct': 7})
        out = r.search('question', top_k=2, candidate_pool=10, stitching_window=1)
        self.assertEqual([c['chunk_id'] for c in out], [1, 10])

    def test_book_filter_excludes_candidates_and_neighbors(self):
        docs = [self.doc(1, 'other', 'Other'), self.doc(2, 'selected')]
        r = self.make(docs, {'other': 10, 'selected': 3})
        out = r.search('question', book_filter='Book A', stitching_window=2)
        self.assertEqual(out[0]['text'], 'selected')
        self.assertEqual(self.seen, [['question', 'selected']])

    def test_source_filter_is_exact(self):
        docs = [self.doc(1, 'other', 'Book A extra'), self.doc(2, 'selected')]
        r = self.make(docs)
        out = r.search('question', source_filter=['Book A'], stitching_window=2)
        self.assertEqual(out[0]['text'], 'selected')

    def test_inspection_exposes_same_final_selection_as_search(self):
        docs = [self.doc(1, 'one'), self.doc(2, 'two')]
        r = self.make(docs, {'one': 2, 'two': -1})
        inspection = r.inspect('question', stitching_window=0)
        self.assertEqual(inspection['selected_hits'], r.search('question', stitching_window=0))
        self.assertEqual(inspection['stitched_hits'], inspection['selected_hits'])
        self.assertEqual(len(inspection['reranked_hits']), 2)

    def test_without_reranker_keeps_stitching_with_same_source_only(self):
        docs = [self.doc(1, 'one'), self.doc(2, 'two', 'Other')]
        r = self.make(docs)
        self.assertEqual(r.search('one', top_k=1, stitching_window=1)[0]['text'], 'one')


if __name__ == '__main__':
    unittest.main(verbosity=2)
