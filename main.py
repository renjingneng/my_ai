from gensim import corpora, models, similarities, downloader
import pprint
from collections import defaultdict
import logging



def run():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    text_corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]
    # Create a set of frequent words
    stoplist = set('for a of the and to in'.split(' '))
    # Lowercase each document, split it by white space and filter out stopwords
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in text_corpus]

    # Count word frequencies

    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
    # pprint.pprint(texts)
    # pprint.pprint(processed_corpus)

    dictionary = corpora.Dictionary(processed_corpus)
    pprint.pprint(dictionary.token2id)

    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
    pprint.pprint(bow_corpus)

    tfidf = models.TfidfModel(bow_corpus)

    # transform the "system minors" string
    # words = "system response".lower().split()
    # print(tfidf[dictionary.doc2bow(words)])

    index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=12)

    query_document = 'human computer'.split()
    query_bow = dictionary.doc2bow(query_document)
    sims = index[tfidf[query_bow]]
    for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
        print(document_number, score)


if __name__ == '__main__':
    run()
