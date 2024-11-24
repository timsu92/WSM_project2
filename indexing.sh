python3 -m pyserini.index.lucene \
    --collection TrecwebCollection \
    --input WT2G \
    --index indexes/collection \
    --generator DefaultLuceneDocumentGenerator \
    --threads $(nproc) \
    --storePositions --storeDocvectors --storeRaw