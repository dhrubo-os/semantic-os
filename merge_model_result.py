import logging
import os
import pathlib
import numpy as np

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

from utils import normalize_values, normalize_values_min_max_scaler, get_mean_result, get_normalized_weighted_linear_result, count_output_list, host_name
from bm25_result import generate_bm25_result
from sbert_result import generate_sbert_result

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download dataset and unzip the dataset
# dataset = "nfcorpus"
# dataset = "dbpedia-entity"
# dataset = "arguana"
# dataset = "trec-covid"
# dataset = "nq"
dataset = "fiqa"
# dataset = "scifact"
# dataset = "scidocs"
# dataset = "quora"
# dataset = "fever"


# result = {}
#
# for i in range(0, len(datasets)):
#     data_path = custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets") + "/" + datasets[i]
#     corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
#     result[datasets[i]] = {}
#     result[datasets[i]]["query_number"] = len(queries)
#     result[datasets[i]]["passage_number"] = len(corpus)
#
#     tempDict = {}
#     for questionID, question in queries.items():
#         tempDict[questionID] = len(question.split())
#     result[datasets[i]]["avg_query_length"] = sum(tempDict.values())/len(tempDict)
#     result[datasets[i]]["med_query_length"] = np.percentile(list(tempDict.values()), 50)
#
#     tempDict2 = {}
#     for passageID, passage in corpus.items():
#         tempDict2[passageID] = len(passage['text'].split())
#     result[datasets[i]]["avg_passage_length"] = sum(tempDict2.values())/len(tempDict2)
#     result[datasets[i]]["med_passage_length"] = np.percentile(list(tempDict2.values()), 50)
#
#
# print(result)
#
# exit(1)

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")



# This k values are being used for BM25 search
tt_k_values = [1, 3, 5, 10, 100, min(9999, len(corpus))]
# k = 20
# print("Printing factor K: ", k)
#
# tt_k_values = [1, 3, 5, 10, k * 20]
#
# print("printing top k values for bm25:", tt_k_values)

# This K values are being used for dense model search
fh_k_values = [1, 3, 5, 10, 100, 250]
# fh_k_values = [1, 3, 5, 10, 20]

# this k values are being used for scoring
k_values = [1, 3, 5, 10, 100]

index_name = dataset



bm25_result, bm25_retriever = generate_bm25_result(index_name, host_name, corpus, queries, initialize=False,
                                                   k_values=tt_k_values, number_of_shards=10)

# print("Printing bm25 questions and answers: \n")
# q_counter = 0
# for qid, doc_dict in bm25_result.items():
#     if q_counter < 10:
#         doc_counter = 0
#         print("Question id: ", qid, "\n")
#         for doc_id, doc_value in doc_dict.items():
#             if doc_counter < 20:
#                 print(doc_id, " : ", doc_value, "\n")
#                 doc_counter += 1
#         q_counter += 1


# #
bm25_norm_result = normalize_values(bm25_result)
# bm25_norm_result = normalize_values_min_max_scaler(bm25_result)

## this is for custom model. In `generate_sbert_result` you can either provide the model name or the file path of the
# model
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/TASB_Covid_Dhrubo"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/tasb_new_scidocs_9"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/trec_covid_tasb_2023"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/new_nfcorpus"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/for_dhrubo_tasb_nfcorpus"
custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/FIQA_final_custom"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/arguana_custom"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/scifact_custom"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/dbpedia_custom_small"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/quora_custom_tasb_larger"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/fever_custom_tasb"
# custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/for_dhrubo_tasb"

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, custom_model_path, fh_k_values,
                                                      batch_size=16)
# sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-roberta-base-ance-firstp", fh_k_values,
#                                                       batch_size=16)
# sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-distilbert-base-tas-b", fh_k_values,
#                                                       batch_size=16)

# print("Printing Dense model questions and answers: \n")
# q_counter = 0
# for qid, doc_dict in sbert_result.items():
#     if q_counter < 10:
#         doc_counter = 0
#         print("Question id: ", qid, "\n")
#         for doc_id, doc_value in doc_dict.items():
#             if doc_counter < 20:
#                 print(doc_id, " : ", doc_value, "\n")
#                 doc_counter += 1
#         q_counter += 1


# count_dense_dict = count_output_list(sbert_result)

# print("number of the question that doesn't have 100 DenseModel results: ", len(count_dense_dict))

sbert_norm_result = normalize_values(sbert_result)
# sbert_norm_result = normalize_values_min_max_scaler(sbert_result)

# merged_result = get_mean_result(bm25_norm_result, sbert_norm_result, meanType="arithmatic")
# merged_result = get_mean_result(bm25_result, sbert_result, meanType="arithmatic")
# arithmatic   geometric   harmonic
# merged_result = get_mean_result(bm25_result, sbert_result, meanType="harmonic")
merged_result = get_normalized_weighted_linear_result(bm25_norm_result, sbert_norm_result, 1)

# print("Printing Merge result questions and answers: \n")
# q_counter = 0
# for qid, doc_dict in merged_result.items():
#     if q_counter < 10:
#         doc_counter = 0
#         print("Question id: ", qid, "\n")
#         for doc_id, doc_value in doc_dict.items():
#             if doc_counter < 20:
#                 print(doc_id, " : ", doc_value, "\n")
#                 doc_counter += 1
#         q_counter += 1



# print("Number of questions:", len(merged_result))
#
# count_dict = count_output_list(merged_result)

# print("number of the question that doesn't have 100 results: ", len(count_dict))

ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, merged_result, k_values)
# ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, merged_result, k_values)

print("Printing ndcg:", ndcg)
print("Printing _map:", _map)
print("Printing precision:", precision)
print("Printing recall:", recall)

#### Retrieval Example ####
# query_id, scores_dict = random.choice(list(merged_result.items()))
#
# logging.info("Query : %s\n" % queries[query_id])
#
# scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
# for rank in range(10):
#     doc_id = scores[rank][0]
#     logging.info("Doc %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))




