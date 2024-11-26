import argparse
from math import log
import os
import itertools
import subprocess

from pyserini.search.lucene import LuceneSearcher
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
from tqdm import tqdm

from proj2_sample_run.sparse_retrieval.codes.util import read_title_desc
from proj2_sample_run.sparse_retrieval.codes.search import search

def evaluate_bm25_params(searcher, query_path, output, args):
    query = read_title_desc(query_path)
    search(searcher, query, output, "bm25", args.k)

    result = subprocess.run(
        ['perl', 'proj2_sample_run/sparse_retrieval/trec_eval.pl',
         'proj2_sample_run/data/qrels.401.txt', output],
        capture_output=True, text=True
    )

    rel_ret = 0
    interpolated_ap = 0
    precision = 0
    r_precision = 0

    stdoutLines = result.stdout.splitlines()
    for lineno, line in enumerate(stdoutLines):
        if "Rel_ret:" in line:
            rel_ret = int(line.split()[-1])
        elif "Interpolated Recall - Precision Averages:" in line:
            for ap_lineno in range(1, 12):
                interpolated_ap += float(
                    stdoutLines[lineno + ap_lineno].split()[-1]
                ) - log(ap_lineno, 400)
        elif "Precision:" in line:
            for p_lineno in range(1, 10):
                precision += float(
                    stdoutLines[lineno + p_lineno].split()[-1]
                ) - log(p_lineno, 400)
        elif "R-Precision " in line:
            r_precision = float(stdoutLines[lineno + 1].split()[-1])

    # score = rel_ret / 45 \
    #         + interpolated_ap / 4 \
    #         + precision / 0.7 \
    #         + r_precision / 0.7

    return rel_ret, interpolated_ap, precision, r_precision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/collection", type=str)
    parser.add_argument("--query", default="proj2_sample_run/data/topics.401.txt", type=str)
    parser.add_argument("--k", default=1000, type=int)
    parser.add_argument("--output_dir", default='runs/bm25_params', type=str)
    parser.add_argument("--log_dir", default='tfboard_logs', type=str)

    args = parser.parse_args()

    writer = SummaryWriter(args.log_dir)

    k1_values = np.arange(1.2, 2.1, 0.1).tolist()
    b_values = np.arange(0.0, 1.1, 0.1).tolist()

    best_score = -(1 << 31)  # min int
    best_params = None

    for i, (k1, b) in tqdm(enumerate(itertools.product(k1_values, b_values))):
        output = os.path.join(args.output_dir, f'bm25_k1_{k1}_b_{b}.run')

        searcher = LuceneSearcher(args.index, "WT2G")
        searcher.set_bm25(k1=k1, b=b)
        scores = evaluate_bm25_params(searcher, args.query, output, args)
        weighted_score = scores[0] / 45 \
                        + scores[1] / 4 \
                        + scores[2] / 0.7 \
                        + scores[3] / 0.7

        writer.add_scalars('BM25', {
            'Rel_ret': scores[0],
            'Interpolated Precision': scores[1],
            'Precision': scores[2],
            'R-Precision': scores[3],
            'Weighted Score': weighted_score,
            'k1': k1,
            'b': b,
        }, global_step=i)

        if weighted_score > best_score:
            best_score = weighted_score
            best_params = (k1, b)

    writer.close()
    print(f"Best params: k1={best_params[0]}, b={best_params[1]} with score {best_score}")

if __name__ == "__main__":
    main()
