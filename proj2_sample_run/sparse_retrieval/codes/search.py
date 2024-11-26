from tqdm import tqdm


def search(searcher, query, output, method, k):
    output = open(output, 'w')

    print(f'Do {method} search...')
    for qid, qtext in tqdm(query.items()):
        hits = searcher.search(qtext, k=k)
        for i in range(len(hits)):
            # trec format: qid Q0 docid rank score method
            output.write(f'{qid} Q0 {hits[i].docid} {i+1} {hits[i].score:.5f} {method}\n')
