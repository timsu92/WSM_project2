import argparse

from pyserini.search.lucene import LuceneSearcher
from pyserini.index import LuceneIndexReader

from proj2_sample_run.sparse_retrieval.codes.util import read_title_desc
from proj2_sample_run.sparse_retrieval.codes.search import search
from searchers.LM_Jelinek_Mercer.searcher import LMJelinekMercerSmoothing

parser = argparse.ArgumentParser()
parser.add_argument("--index", default="indexes/collection", type=str)
parser.add_argument("--query", default="proj2_sample_run/data/topics.401.txt", type=str)
parser.add_argument("--method", default="bm25", type=str, choices=("bm25", "MLE_laplace", "LM_JM"))
parser.add_argument("--k", default=1000, type=int)
parser.add_argument("--output", default='runs/bm25.run', type=str)

args = parser.parse_args()


if args.method == "bm25":
    searcher = LuceneSearcher(args.index, prebuilt_index_name="WT2G")
    # searcher.set_bm25(k1=1.8, b=0.4)
    searcher.set_bm25(k1=2, b=0.75)
elif args.method == "MLE_laplace":  # laplace smoothing
    searcher = LuceneSearcher(args.index, prebuilt_index_name="WT2G")
    index_reader = LuceneIndexReader(args.index)
    searcher.set_qld(index_reader.stats()["unique_terms"])
elif args.method == "LM_JM":
    searcher = LMJelinekMercerSmoothing(index_dir=args.index, prebuilt_index_name="WT2G")

query = read_title_desc(args.query)
search(searcher, query, args.output, args.method, args.k)
