import argparse
from collections import defaultdict
import os
from time import sleep

from pyserini.search.lucene import LuceneSearcher
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from searchers.LM_Jelinek_Mercer.searcher import LMJelinekMercerSmoothing
from proj2_sample_run.sparse_retrieval.codes.util import read_title_desc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RetrievalDataset(Dataset):
    def __init__(self, data):
        """
        data: List of tuples (qid, docid, [bm25_score, mle_laplace_score, lm_jm_score], label)
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, _, scores, label = self.data[idx]
        return torch.tensor(scores, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

bm25Searcher = LuceneSearcher("indexes/collection", "WT2G")
bm25Searcher.set_bm25(k1=2, b=0.75)

MLE_laplaceSearcher = LuceneSearcher("indexes/collection", "WT2G")
# index_reader = LuceneIndexReader("indexes/collection")
# MLE_laplaceSearcher.set_qld(index_reader.stats()["unique_terms"])
MLE_laplaceSearcher.set_qld(1000)

LM_JM_Searcher = LMJelinekMercerSmoothing(index_dir="indexes/collection", prebuilt_index_name="WT2G")
def build_dataset(queries, answers, k=1000):
    dataset = []
    for qid, qtext in tqdm(queries.items()):
        bm25_hits = bm25Searcher.search(qtext, k=k)
        sleep(0.01)
        mle_hits = MLE_laplaceSearcher.search(qtext, k=k)
        sleep(0.01)
        lmjm_hits = LM_JM_Searcher.search(qtext, k=k)

        doc_scores = defaultdict(lambda: [0.0, 0.0, 0.0])
        # max of scores: 27.19390
        for hit in bm25_hits:
            doc_scores[hit.docid][0] = hit.score
        for hit in mle_hits:
            doc_scores[hit.docid][1] = hit.score
        for hit in lmjm_hits:
            doc_scores[hit.docid][2] = hit.score

        for docid, scores in doc_scores.items():
            label = answers[qid].get(docid, 0)
            dataset.append((qid, docid, scores, label))
    return dataset

### classifier

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x).squeeze()
                loss = criterion(y_pred, y)
                val_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            y_pred = (y_pred > 0.5).int()  # 閾值設定為 0.5，轉換為 0 或 1
            all_preds.extend(y_pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_query", default="proj2_sample_run/data/topics.401-440.txt", type=str)
    parser.add_argument("--train_answer", default="proj2_sample_run/data/qrels.401-440.txt", type=str)
    parser.add_argument("--test_query", default="proj2_sample_run/data/topics.441-450.txt", type=str)
    parser.add_argument("--test_answer", default="proj2_sample_run/data/qrels.441-450.txt", type=str)
    parser.add_argument("--output_dir", default='runs/part2', type=str)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)

    args = parser.parse_args()

    # train set
    queries = read_title_desc(args.train_query)
    with open(args.train_answer, 'r') as f:
        answers = defaultdict(lambda: defaultdict(lambda: 0))
        for line in f:
            line_split = line.strip().split()
            qid, docid, rel = line_split[0], line_split[2], int(line_split[3])
            if rel == 1:
                answers[qid][docid] = 1
    print("building train dataset...")
    train_data = build_dataset(queries, answers)

    # test set
    queries = read_title_desc(args.test_query)
    with open(args.test_answer, 'r') as f:
        answers = defaultdict(lambda: defaultdict(lambda: 0))
        for line in f:
            line_split = line.strip().split()
            qid, docid, rel = line_split[0], line_split[2], int(line_split[3])
            if rel == 1:
                answers[qid][docid] = 1
    print("building test dataset...")
    test_data = build_dataset(queries, answers)

    train_loader = DataLoader(RetrievalDataset(train_data), batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() or 8)
    test_loader = DataLoader(RetrievalDataset(test_data), batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() or 8)

    model = BinaryClassifier(input_dim=3)  # (bm25, mle_laplace, lm_jm)
    train_model(model, train_loader, test_loader, num_epochs=args.num_epochs, lr=args.lr)
    torch.save(model.state_dict(), 'model.pth')

    evaluate_model(model, test_loader)