import argparse

from pyserini.search.lucene import LuceneSearcher

from proj2_sample_run.sparse_retrieval.codes.util import read_desc, read_title, read_title_desc
from proj2_sample_run.sparse_retrieval.codes.search import search

parser = argparse.ArgumentParser()
parser.add_argument("--index", default="indexes/collection", type=str)
parser.add_argument("--query", default="proj2_sample_run/data/topics.401.txt", type=str)
parser.add_argument("--method", default="bm25", type=str)
parser.add_argument("--k", default=1000, type=int)
parser.add_argument("--output", default='runs/bm25.run', type=str)

args = parser.parse_args()


searcher = LuceneSearcher(args.index, "WT2G")
if args.method == "bm25":
    # searcher.set_bm25(k1=1.8, b=0.4)
    searcher.set_bm25(k1=2, b=0.75)

query = read_title_desc(args.query)
search(searcher, query, args.output, args.method, args.k)
