python -m pyserini.index.lucene \
  --collection TrecwebCollection \
  --input ../data/WT2G \
  --index indexes/collection \
  --generator DefaultLuceneDocumentGenerator \
  --threads 14 \
  --storePositions --storeDocvectors --storeRaw
