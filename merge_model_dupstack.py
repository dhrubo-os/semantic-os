from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
import pathlib, os
from utils import normalize_values, get_mean_result, host_name
from bm25_result import generate_bm25_result
from sbert_result import generate_sbert_result
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "cqadupstack"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path("__file__").parent.absolute(), "datasets")
print("Printing out directory:", out_dir)

data_path = util.download_and_unzip(url, out_dir)

custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/dupstack_custom_tasb"

corpus = {}
queries = {}
qrels = {}
mega_corpus = []
mega_queries = []
mega_qrels = []
data_name = []
total_len_corp = 0
total_len_q = 0
total_len_qrels = 0
for filename in os.listdir(data_path):
    f = os.path.join(data_path, filename) + "/"
    print(f)
    corpus, queries, qrels = GenericDataLoader(data_folder=f).load(split="test")
    mega_corpus.append(corpus)
    mega_qrels.append(qrels)
    mega_queries.append(queries)
    data_name.append(filename)

ndcg_list = []
recall_list = []

fh_k_values = [1, 3, 5, 10, 100, 250]

# this k values are being used for scoring
k_values = [1, 3, 5, 10, 100]


result = {}
query_no = 0
passage_no = 0

word_no_query = 0
word_no_passage = 0

tempDict1 = {}
tempDict2 = {}

# for i in range(len(mega_corpus)):
#     passage_no = passage_no + len(mega_corpus[i])
#     query_no = query_no + len(mega_queries[i])
#
#     for questionID, question in mega_queries[i].items():
#         print("printing question::", question)
#         word_no_query = word_no_query + len(question.split())
#         tempDict1[questionID] = len(question["body"].split())
#
#     for passageID, passage in mega_corpus[i].items():
#         word_no_passage = word_no_passage + len(passage['text'].split())
#         tempDict2[questionID] = len(passage['text'].split())
#
# print("query number: ", query_no)
# print("passage number: ", passage_no)
# print("avg query length: ", word_no_query/query_no)
# print("avg passage length: ", word_no_passage/passage_no)
# print("med query length: ", np.percentile(list(tempDict1.values()), 50))
# print("med passage length: ", np.percentile(list(tempDict2.values()), 50))

for i in range(len(mega_corpus)):
    print(i)
    print("Experiment for ", data_name[i], " is running")

    tt_k_values = [1, 3, 5, 10, 100, min(9999, len(mega_corpus[i]))]
    index_name = "cqadupstack_" + data_name[i]
    bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, mega_corpus[i], mega_queries[i], initialize=True,
                                                       k_values=tt_k_values, number_of_shards=10)
    #
    bm25_norm_result = normalize_values(bm25_result)

    sbert_result, dense_retriever = generate_sbert_result(mega_corpus[i], mega_queries[i], custom_model_path, fh_k_values,
    #                                                       batch_size=16)
    # sbert_result, dense_retriever = generate_sbert_result(mega_corpus[i], mega_queries[i], "msmarco-distilbert-base-tas-b", fh_k_values,
                                                          batch_size=16)
    #
    sbert_norm_result = normalize_values(sbert_result)
    # # arithmatic   geometric   harmonic
    merged_result = get_mean_result(bm25_norm_result, sbert_norm_result, meanType="geometric")
    # merged_result = get_mean_result(bm25_result, sbert_result, meanType="arithmatic")

    ndcg, _map, recall, precision = dense_retriever.evaluate(mega_qrels[i], merged_result, k_values)

    print("Printing results for: ", data_name[i])
    print("Printing ndcg:", ndcg)
    print("Printing _map:", _map)
    print("Printing precision:", precision)
    print("Printing recall:", recall)

    # custom_model = DRES(YourCustomDEModel(model, model.tokenizer))
    # retriever_model = EvaluateRetrieval(custom_model, score_function="dot")
    # results = retriever_model.retrieve(corpus, mega_queries[i])
    # ndcg, _map, recall, precision = retriever_model.evaluate(mega_qrels[i], results, retriever_model.k_values)
    # ndcg_list.append(ndcg["NDCG@10"])
    # recall_list.append(recall["Recall@100"])
# ndcg = np.average(ndcg_list)
# recall = np.average(recall_list)