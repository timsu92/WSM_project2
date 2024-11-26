import argparse
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from searchers.LM_Jelinek_Mercer.searcher import LMJelinekMercerSmoothing
from proj2_sample_run.sparse_retrieval.codes.util import read_title_desc

def build_scores(queries, searcher, k=1000):
    results = defaultdict(lambda: defaultdict(float))
    for qid, qtext in tqdm(queries.items()):
        hits = searcher.search(qtext, k=k)
        for hit in hits:
            results[qid][hit.docid] = hit.score
    return results

def evaluate_searcher(scores, answers):
    all_preds = []
    all_labels = []

    for qid, docs in answers.items():
        for docid, label in docs.items():
            score = scores[qid].get(docid, 0)
            pred = 1 if score > 12 else 0  # 因為最大找到的分數是 27.19390，所以閾值設定為 12
            all_preds.append(pred)
            all_labels.append(label)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_query", default="proj2_sample_run/data/topics.441-450.txt", type=str)
    parser.add_argument("--test_answer", default="proj2_sample_run/data/qrels.441-450.txt", type=str)
    args = parser.parse_args()

    # 讀取測試集
    queries = read_title_desc(args.test_query)
    with open(args.test_answer, 'r') as f:
        answers = defaultdict(lambda: defaultdict(lambda: 0))
        for line in f:
            line_split = line.strip().split()
            qid, docid, rel = line_split[0], line_split[2], int(line_split[3])
            if rel == 1:
                answers[qid][docid] = 1

    # 初始化檢索器
    bm25Searcher = LuceneSearcher("indexes/collection", "WT2G")
    bm25Searcher.set_bm25(k1=2, b=0.75)

    MLE_laplaceSearcher = LuceneSearcher("indexes/collection", "WT2G")
    MLE_laplaceSearcher.set_qld(1000)

    LM_JM_Searcher = LMJelinekMercerSmoothing(index_dir="indexes/collection", prebuilt_index_name="WT2G")

    # 評估 BM25
    print("Evaluating BM25...")
    bm25_scores = build_scores(queries, bm25Searcher)
    bm25_metrics = evaluate_searcher(bm25_scores, answers)
    print("BM25 Evaluation Metrics:")
    print(f"\tAccuracy:  {bm25_metrics[0]:.4f}")
    print(f"\tPrecision: {bm25_metrics[1]:.4f}")
    print(f"\tRecall:    {bm25_metrics[2]:.4f}")
    print(f"\tF1 Score:  {bm25_metrics[3]:.4f}")

    # 評估 MLE Laplace
    print("\nEvaluating MLE Laplace...")
    mle_laplace_scores = build_scores(queries, MLE_laplaceSearcher)
    mle_laplace_metrics = evaluate_searcher(mle_laplace_scores, answers)
    print("MLE Laplace Evaluation Metrics:")
    print(f"\tAccuracy:  {mle_laplace_metrics[0]:.4f}")
    print(f"\tPrecision: {mle_laplace_metrics[1]:.4f}")
    print(f"\tRecall:    {mle_laplace_metrics[2]:.4f}")
    print(f"\tF1 Score:  {mle_laplace_metrics[3]:.4f}")

    # 評估 LM Jelinek Mercer
    print("\nEvaluating LM Jelinek Mercer...")
    lm_jm_scores = build_scores(queries, LM_JM_Searcher)
    lm_jm_metrics = evaluate_searcher(lm_jm_scores, answers)
    print("LM Jelinek Mercer Evaluation Metrics:")
    print(f"\tAccuracy:  {lm_jm_metrics[0]:.4f}")
    print(f"\tPrecision: {lm_jm_metrics[1]:.4f}")
    print(f"\tRecall:    {lm_jm_metrics[2]:.4f}")
    print(f"\tF1 Score:  {lm_jm_metrics[3]:.4f}")
