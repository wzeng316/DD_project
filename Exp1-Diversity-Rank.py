import json

from Exp_model import  *


# Truth = json.load(open('/users/zengwei/data/DD/ZW_JSON/Truth.json'))
# print 'load 1'
# QueryDocs = json.load(open('/users/zengwei/data/DD/ZW_JSON/MiniData_QueryDocs.json'))
# print 'load 3'
# Doc2Vec = json.load(open('/users/zengwei/data/DD/ZW_JSON/MiniData_200_50ite_Doc2Vec.json'))
# print 'load 4'
#
# Nfeature=200
# Nhidden = 10
# Lenepisode = 50
# lr=0.01
#
# model = DiverseRank(Lenepisode, Nfeature, Nhidden, lr)
#
# model.main(1000, QueryDocs, Doc2Vec, Truth)




# Truth = json.load(open('/users/zengwei/data/DD/ZW_JSON/Truth.json'))
# print 'load 1'
# QueryDocs = json.load(open('/users/zengwei/data/DD/ZW_JSON/All_QueryDocs_3K.json'))
# print 'load 3'
# Doc2Vec = json.load(open('/users/zengwei/data/DD/ZW_JSON/All_200_50ite_Doc2Vec.json'))
# print 'load 4'
# Nfeature=200
# Nhidden = 10
# Lenepisode = 50
# lr=0.01
# model = DiverseRank(Lenepisode, Nfeature, Nhidden, lr)
# model.main_3K(1000, QueryDocs, Doc2Vec, Truth)



Truth = json.load(open('/users/zengwei/data/DD/ZW_JSON/Truth.json'))
print 'load 1'
QueryDocs = json.load(open('/users/zengwei/data/DD/ZW_JSON/All_QueryDocs.json'))
print 'load 3'
Doc2Vec = json.load(open('/users/zengwei/data/DD/ZW_JSON/All_200_50ite_Doc2Vec.json'))
print 'load 4'

Nfeature=200
Nhidden = 10
Lenepisode = 50
lr=0.1

model = DiverseRank(Lenepisode, Nfeature, Nhidden, lr)

model.main_3KCV(10000, QueryDocs, Doc2Vec, Truth)