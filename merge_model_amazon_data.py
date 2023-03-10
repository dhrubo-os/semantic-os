import logging
import os
import pathlib
import numpy as np
import pandas as pd

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

from utils import normalize_values, normalize_values_min_max_scaler, get_mean_result, \
    get_normalized_weighted_linear_result, count_output_list, host_name
from bm25_result import generate_bm25_result
from sbert_result import generate_sbert_result

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


df_examples = pd.read_parquet('datasets/amazon/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('datasets/amazon/shopping_queries_dataset_products.parquet')
# df_sources = pd.read_csv("datasets/amazon/shopping_queries_dataset_sources.csv")

df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale', 'product_id'],
    right_on=['product_locale', 'product_id']
)

# df_task_1 = df_examples_products[df_examples_products["small_version"] == 1]
# df_task_1_train = df_task_1[df_task_1["split"] == "train"]
# df_task_1_test = df_task_1[df_task_1["split"] == "test"]

# df_task_2 = df_examples_products[df_examples_products["large_version"] == 1]
# df_task_2_train = df_task_2[df_task_2["split"] == "train"]
# df_task_2_test = df_task_2[df_task_2["split"] == "test"]

# df_task_3 = df_examples_products[df_examples_products["large_version"] == 1]
# df_task_3["subtitute_label"] = df_task_3["esci_label"].apply(lambda esci_label: 1 if esci_label == "S" else 0)
# del df_task_3["esci_label"]
# df_task_3_train = df_task_3[df_task_3["split"] == "train"]
# df_task_3_test = df_task_3[df_task_3["split"] == "test"]

# df_examples_products_source = pd.merge(
#     df_examples_products,
#     df_sources,
#     how='left',
#     left_on=['query_id'],
#     right_on=['query_id']
# )

df_examples_products_us = df_examples_products[df_examples_products.product_locale == "us"].copy()

df_examples_products_us["bullet_des"] = df_examples_products_us["product_bullet_point"].fillna(" ") + \
                                        df_examples_products_us["product_description"].fillna(" ")

df_examples_products_us["data"] = df_examples_products_us["product_color"].fillna(" ") + \
                                  df_examples_products_us["product_title"].fillna(" ") + \
                                  df_examples_products_us["bullet_des"]

df_task_1_us = df_examples_products_us[df_examples_products_us["small_version"] == 1]
df_task_1_train_us = df_task_1_us[df_task_1_us["split"] == "train"]
df_task_1_test_us = df_task_1_us[df_task_1_us["split"] == "test"]

i = np.random.randint(len(df_task_1_test_us))
df_task_1_test_us[i:i + 10]

# tot_queries = {}
# for index, row in df_task_1.iterrows():
#     if row["query_id"] not in tot_queries:
#         tot_queries[row["query_id"]] = row["query"]

val_legend = {"E": 100, "S": 10, "C": 1, "I": 0}

print("Preparing queries")
queries = {}
for index, row in df_task_1_test_us.iterrows():
    if row["query_id"] not in queries:
        queries[str(row["query_id"])] = row["query"]

print("Preparing qrels")
qrels = {}
for index, row in df_task_1_test_us.iterrows():
    if row["query_id"] not in qrels:
        qrels[str(row["query_id"])] = {row["product_id"]: val_legend[row["esci_label"]]}
    else:
        qrels[str(row["query_id"])].update({row["product_id"]: val_legend[row["esci_label"]]})

print("Preparing corpus")
corpus = {}
for index, row in df_task_1_us.iterrows():
    if row["product_id"] not in corpus:
        corpus[str(row["product_id"])] = {"text": row["data"]}

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

index_name = "amazon_esci"

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
custom_model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "models") + "/tasb_amazon_new_19"

# this section takes a lot of time as this generates embedding for questions and answers and then finds the
# similary between both embedded values
# sbert_result, dense_retriever = generate_sbert_result(corpus, queries, custom_model_path, fh_k_values,
#                                                       batch_size=16)
# sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-roberta-base-ance-firstp", fh_k_values,
#                                                       batch_size=16)
sbert_result, dense_retriever = generate_sbert_result(corpus, queries, "msmarco-distilbert-base-tas-b", fh_k_values,
                                                       batch_size=16)

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

# merged_result = get_mean_result(bm25_norm_result, sbert_norm_result, meanType="harmonic")
# merged_result = get_mean_result(bm25_result, sbert_result, meanType="harmonic")
# arithmatic   geometric   harmonic
# merged_result = get_mean_result(bm25_result, sbert_result, meanType="harmonic")
merged_result = get_normalized_weighted_linear_result(bm25_norm_result, sbert_norm_result, 1024)

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

# ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, sbert_result, k_values)
ndcg, _map, recall, precision = dense_retriever.evaluate(qrels, merged_result, k_values)

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
