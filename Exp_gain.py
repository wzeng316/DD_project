from truth import *
from reader import *
from collections import Counter
import numpy as np
import statistics
import argparse
import sys


########################################################################################################################
# Cube Test Gain
########################################################################################################################

def CT_gain_per_doc(doc, topic_truth, subtopic_height, subtopic_count, weight_per_subtopic, max_height=1, gamma=0.5):
    if doc not in topic_truth.keys():
        return 0

    gain = 0
    for subtopic_id, rating in topic_truth[doc].items():
        if subtopic_height[subtopic_id] < max_height:
            discount_height = (gamma ** (subtopic_count[subtopic_id] + 1)) * rating
            if discount_height + subtopic_height[subtopic_id] > max_height:
                discount_height = max_height - subtopic_height[subtopic_id]

            gain += weight_per_subtopic * discount_height
            # print(doc_no, subtopic_id,"original_height", rating, "discount height", discount_height)
            subtopic_height[subtopic_id] += discount_height
            subtopic_count[subtopic_id] += 1
    # print(doc_no, gain)
    return gain


def CT_gain_per_query(doc_list, topic_truth, max_height=1, gamma=0.5):

    subtopic_num = topic_truth[1]
    topic_truth  = topic_truth[0]

    subtopic_height = Counter()
    subtopic_count  = Counter()
    weight_per_subtopic =1.0/subtopic_num

    time = 0.0
    gain=[]
    for doc in doc_list:
        time += 1
        doc_gain = CT_gain_per_doc(doc, topic_truth, subtopic_height, subtopic_count, weight_per_subtopic, max_height, gamma)
        gain.append(doc_gain / max_height / time)

    return gain


def CT_get_reward(doc_list, topic_truth, max_height=1, gamma = 0.5):
    reward = CT_gain_per_query(doc_list, topic_truth, max_height, gamma)
    ndoc = len(reward)

    for i in range(ndoc - 1):
        docid = ndoc - 2 - i
        reward[docid] += reward[docid + 1]
    return reward




########################################################################################################################
# aloha DCG
########################################################################################################################

def alphaDCG_gain_per_doc(doc, topic_truth, subtopic_count, alpha=0.5):

    if doc not in topic_truth.keys():
        return 0

    gain = 0

    subtopic=[]
    for subtopic_id, _ in topic_truth[doc]:

        if subtopic_id in subtopic:
            continue

        if subtopic_count[subtopic_id] == 0:
            gain += 1 - alpha
        else:
            gain += (1 - alpha) ** subtopic_count[subtopic_id]
        subtopic.append(subtopic_id)
        subtopic_count[subtopic_id] += 1
    return gain


def alphaDCG_gain_per_query(doc_list, topic_truth, alpha=0.5):

    subtopic_count = Counter()
    topic_truth = topic_truth[0]

    time = 0.0
    gain = []
    for doc in doc_list:
        doc_gain = alphaDCG_gain_per_doc(doc, topic_truth, subtopic_count, alpha)
        gain.append(doc_gain / np.math.log(time + 2, 2))
        time += 1
    return gain


def alphaDCG_reward_per_query(doc_list, topic_truth, alpha = 0.5):
    reward = alphaDCG_gain_per_query(doc_list, topic_truth, alpha)
    ndoc = len(reward)

    for i in range(ndoc - 1):
        docid = ndoc - 2 - i
        reward[docid] += reward[docid + 1]
    return reward

def max_alphaDCG_per_doc(doc_list, topic_truth, subtopic_bound_count, alpha):
    doc_gain =0
    doc_id =0
    ndoc = len(doc_list)
    for docid in xrange(ndoc):
        doc = doc_list[docid]

        if doc not in topic_truth.keys():
            continue
        gain =0
        subtopic = []
        for subtopic_id, _ in topic_truth[doc]:
            if subtopic_id in subtopic:
                continue
            if subtopic_bound_count[subtopic_id] == 0:
                gain += 1 - alpha
            else:
                gain += (1 - alpha) ** subtopic_bound_count[subtopic_id]
            subtopic.append(subtopic_id)

        if gain > doc_gain:
            doc_gain=gain
            doc_id = docid

    if doc_gain>0:
        doc = doc_list[doc_id]
        subtopic = []
        for subtopic_id, _ in topic_truth[doc]:
            if subtopic_id not in subtopic:
                subtopic.append(subtopic_id)
                subtopic_bound_count[subtopic_id]+=1

    return  doc_id, doc_gain



def alphaNDCG_per_query(doc_list, topic_truth, top_n=10, alpha=0.5):

    topic_truth = topic_truth[0]

    subtopic_count=Counter()
    subtopic_bound_count=Counter()

    time = 0.0
    gain = []
    for doc in doc_list[ 0 : top_n]:
        doc_gain = alphaDCG_gain_per_doc(doc, topic_truth, subtopic_count, alpha)
        gain.append(doc_gain / np.math.log(time + 2, 2))
        time += 1.0

    time = 0.0
    max_gain=[]
    doc_list_tmp = doc_list[:]
    for idx in range(top_n):
        doc_id, doc_gain = max_alphaDCG_per_doc(doc_list_tmp, topic_truth, subtopic_bound_count, alpha)
        max_gain.append(doc_gain/np.math.log(time+2,2))
        time += 1.0
        del doc_list_tmp[doc_id]

    alpha_NDCG = []
    for i in xrange(top_n):
        if i == 0:
            if max_gain[i]== 0:
                alpha_NDCG.append(0)
            else:
                alpha_NDCG.append(gain[i]/max_gain[i])
        else:
            gain[i] += gain[i-1]
            max_gain[i] += max_gain[i-1]

            if max_gain[i] == 0:
                alpha_NDCG.append(0)
            else:
                alpha_NDCG.append(gain[i]/max_gain[i])
    return alpha_NDCG


def alphaDCG_group_gain_per_query(doc_list, topic_truth, alpha=0.5, group_size=5):

    subtopic_count = Counter()
    topic_truth = topic_truth[0]

    n_group = len(doc_list)/group_size

    gain = []

    doc_id =0
    for i in range(n_group):
        group_gain=0.0
        for j in range(group_size):
            doc = doc_list[doc_id]
            doc_gain = alphaDCG_gain_per_doc(doc, topic_truth, subtopic_count, alpha)
            group_gain += doc_gain / np.math.log(doc_id + 2, 2)
            doc_id += 1
        gain.append(group_gain)
    return gain


def alphaDCG_group_reward_per_query(doc_list, topic_truth, alpha = 0.5, group_size=5):
    reward = alphaDCG_group_gain_per_query(doc_list, topic_truth, alpha, group_size)

    ndoc = len(reward)
    for i in range(ndoc - 1):
        docid = ndoc - 2 - i
        reward[docid] += reward[docid + 1]
    return reward



