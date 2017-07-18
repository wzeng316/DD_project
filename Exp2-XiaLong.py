import json

import yaml

from Exp_model import  *

QueryDocs_Xia = json.load(open('/home/zw/data/XiaLong/query_doc.json'))

QueryDocs = {}
for query in QueryDocs_Xia.keys():

    QueryDocs[query] = []
    for doc in QueryDocs_Xia[query].keys():
        QueryDocs[query].append(doc)
print 'load 2'

QueryDocumentTopic = {}
for query in QueryDocs_Xia.keys():
    QueryDocumentTopic[query] = {}

    for doc in QueryDocs_Xia[query].keys():
        if len(QueryDocs_Xia[query][doc]) > 0:
            QueryDocumentTopic[query][doc] = QueryDocs_Xia[query][doc]
print 'load 3'

Truth = {}
for query in QueryDocs_Xia.keys():
    Truth[query] = []
    Truth[query].append({})
    subtopic_list = []
    for doc in QueryDocs_Xia[query].keys():
        if len(QueryDocs_Xia[query][doc]) > 0:
            Truth[query][0][doc] = []
            for subtopic in QueryDocs_Xia[query][doc]:
                Truth[query][0][doc].append((subtopic, 1))
                if subtopic not in subtopic_list:
                    subtopic_list.append(subtopic)

    Truth[query].append(len(subtopic_list))



Doc2Vec_doc   = json.load(open('/home/zw/data/XiaLong/doc_representation.dat'))
Doc2Vec_query = json.load(open('/home/zw/data/XiaLong/query_representation.dat'))
Doc2Vec =dict (Doc2Vec_doc, **Doc2Vec_query)
print 'load 4'


#######################################################################################################################
Nfeature=100
Nhidden = 10
Lenepisode = 10
lr=0.1

model = DiverseRank(Lenepisode, Nfeature, Nhidden, lr)


model.main(10000, QueryDocs, Doc2Vec, Truth)
