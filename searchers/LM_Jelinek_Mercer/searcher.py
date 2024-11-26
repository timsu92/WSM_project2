from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import autoclass

LMJelinekMercerSimilarity = autoclass('org.apache.lucene.search.similarities.LMJelinekMercerSimilarity')
IndexSearcher = autoclass('org.apache.lucene.search.IndexSearcher')

class LMJelinekMercerSmoothing(LuceneSearcher):
    def __init__(self, lambda_of_jm: float = 0.8, *qrgs, **kwargs):
        super().__init__(*qrgs, **kwargs)
        self.object.similarity = LMJelinekMercerSimilarity(lambda_of_jm)  # inverse of what's given in wm5
        self.object.searcher = IndexSearcher(self.object.reader)
        self.object.searcher.setSimilarity(self.object.similarity)