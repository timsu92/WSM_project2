import argparse
from collections import defaultdict
from html import parser
from time import sleep

import torch
from tqdm import tqdm

from proj2_sample_run.sparse_retrieval.codes.util import read_title_desc
from queries_part2.train import device, BinaryClassifier, bm25Searcher, MLE_laplaceSearcher, LM_JM_Searcher

def predict_and_write_trec(model, queries, searchers, output_file, k=1000):
    model.eval()
    model.to(device)
    with open(output_file, 'w') as output:
        for qid, qtext in tqdm(queries.items()):
            # 收集每個檢索器的分數
            doc_scores = defaultdict(lambda: [0.0, 0.0, 0.0])

            for i, searcher in enumerate(searchers):
                hits = searcher.search(qtext, k=k)
                sleep(0.001)
                for hit in hits:
                    doc_scores[hit.docid][i] = hit.score

            # 使用模型進行推論
            results = []
            with torch.no_grad():
                for docid, scores in doc_scores.items():
                    scores_tensor = torch.tensor(scores, dtype=torch.float32).to(device)
                    relevance = model(scores_tensor).item()
                    results.append((docid, relevance))

            # 根據推論分數排序並寫入 TREC 格式
            results.sort(key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(results, start=1):
                output.write(f"{qid} Q0 {docid} {rank} {score:.5f} MYMODEL\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_query", default="proj2_sample_run/data/topics.441-450.txt", type=str)
    parser.add_argument("--output", default='eval_part2/mymodel.run', type=str)
    parser.add_argument("--model_path", default='model.pth', type=str)

    args = parser.parse_args()

    # 讀取測試集
    queries = read_title_desc(args.test_query)

    model = BinaryClassifier(input_dim=3)  # (bm25, mle_laplace, lm_jm)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))

    predict_and_write_trec(model, queries, [bm25Searcher, MLE_laplaceSearcher, LM_JM_Searcher], args.output)
