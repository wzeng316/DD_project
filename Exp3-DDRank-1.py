import json
import yaml


from Exp_model import  *



Truth = json.load(open('/home/zw/DD/data/Truth.json'))
print 'load 1'
IdQuery = json.load(open('/home/zw/DD/data/IdQuery.json'))
print 'load 2'
QueryDocs = json.load(open('/home/zw/DD/data/QueryDocs.json'))
print 'load 3'
Doc2Vec = yaml.load(open('/home/zw/DD/data/Doc2Vec.yml'))
print 'load 4'


Nfeature=200
Nhidden = 100
Lenepisode = 10
lr=0.01

model = DDRank_1(Lenepisode, Nfeature, Nhidden, lr)


model.main(100000, QueryDocs, Doc2Vec, Truth)
