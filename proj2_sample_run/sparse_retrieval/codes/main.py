import argparse
from search import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/collection", type=str)
    parser.add_argument("--query", default="../data/topics.401.txt", type=str)
    parser.add_argument("--method", default="bm25", type=str)
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--output", default='runs/bm25.run', type=str)

    args = parser.parse_args()

    searcher = LuceneSearcher(args.index)
    if args.method == "bm25":
        searcher.set_bm25(k1=2, b=0.75)

    query = read_desc(args.query)
    search(searcher, query, args)
