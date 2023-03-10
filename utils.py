from collections import defaultdict
from statistics import harmonic_mean, mean, geometric_mean

import numpy as np
from sklearn import preprocessing

host_name = "localhost"

def normalize_values(seach_result):
    '''
         This method normalizes values of a nested dictionary with in-place replacement.

         @param seach_result:        dict     a nested dictionary of question id and dictionary of document id and score

         @return seach_result:       dict     a nested dictionary of question id and dictionary of document id and
                                              normalized score
    '''
    for _, d in seach_result.items():
        x_array = np.array(list(d.values())).astype(float)
        normalized_arr = preprocessing.normalize([x_array])
        normalized_arr = normalized_arr.tolist()[0]

        # total_sum = sum(d.values())
        # factor = 1.0 / total_sum
        d.update(zip(d, normalized_arr))

    return seach_result

def normalize_values_min_max_scaler(seach_result):
    '''
         This method normalizes (using min max Scaler) values of a nested dictionary with in-place replacement.

         @param seach_result:        dict     a nested dictionary of question id and dictionary of document id and score

         @return seach_result:       dict     a nested dictionary of question id and dictionary of document id and
                                              normalized score
    '''
    for _, d in seach_result.items():
        x_array = np.array(list(d.values())).astype(float)
        x_array_list = x_array.reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        normalized_list = scaler.fit_transform(x_array_list)
        normalized_list = normalized_list.reshape(len(x_array))

        # total_sum = sum(d.values())
        # factor = 1.0 / total_sum
        d.update(zip(d, normalized_list))

    return seach_result

def get_mean_result(bm25_result, sbert_result, meanType="harmonic"):
    '''
         This method calculates the mean of the search result from bm25 and bert model

         @param bm25_result:        dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score
                                              search result of bm25

         @param sbert_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score
                                              search result of bert transformer

         @param final_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              mean score of both results

    '''
    print(meanType + " is calculating")

    final_result = defaultdict()
    for question_id, doc_dict in sbert_result.items():
        for doc_id, doc_value in doc_dict.items():
            if question_id in bm25_result.keys() and doc_id in bm25_result[question_id].keys():
                if question_id not in final_result.keys():
                    final_result[question_id] = {}
                if meanType == "arithmatic":
                    final_result[question_id][doc_id] = mean([bm25_result[question_id][doc_id],
                                                              sbert_result[question_id][doc_id]])
                elif meanType == "geometric":
                    if bm25_result[question_id][doc_id] > 0 and sbert_result[question_id][doc_id] > 0:
                        final_result[question_id][doc_id] = geometric_mean([bm25_result[question_id][doc_id],
                                                                            sbert_result[question_id][doc_id]])
                else:
                    final_result[question_id][doc_id] = harmonic_mean([bm25_result[question_id][doc_id],
                                                                       sbert_result[question_id][doc_id]])

    return final_result

def get_normalized_weighted_linear_result(bm25_result, sbert_result, factor=1.0):
    '''
         Instead of normalizing the scores, we can just try different weights for each score and combine them linearly.
         score = sbert_result * factor + bm25_result

         @param bm25_result:        dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bm25

         @param sbert_result:      dict      a nested dictionary of question id and dictionary of document id and
                                              normalized score, search result of bert transformer

         @param factor:            number    weight factor to be added with sbert result

         @return final_result:     dict      a nested dictionary of question id and dictionary of document id and
                                              weighted linear result

    '''
    print("factor: ", factor)

    final_result = defaultdict()
    for question_id, doc_dict in sbert_result.items():
        for doc_id, doc_value in doc_dict.items():
            if question_id not in final_result.keys():
                final_result[question_id] = {}
            if question_id in bm25_result.keys() and doc_id in bm25_result[question_id].keys():
                final_result[question_id][doc_id] = bm25_result[question_id][doc_id] + \
                                                    (factor * sbert_result[question_id][doc_id])
            else:
                final_result[question_id][doc_id] = 0 + (factor * sbert_result[question_id][doc_id])

    return final_result

def count_output_list(merged_result):
    count_dict = {}
    for question_id, doc_dict in merged_result.items():
        if len(doc_dict) < 100:
            count_dict[question_id] = len(doc_dict)
    return count_dict