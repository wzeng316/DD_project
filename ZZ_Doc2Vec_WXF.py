import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import yaml

class TaggedResultsDocuments(object):
    def __init__(self, datatype):

        self.index = range(1, 54)

        # Create a set of frequent words
        self.stoplist = set()
        for word in open('/home/zengwei/DD/data/stoplist', 'r').readlines():
            self.stoplist.add(word.strip('\n'))
        self.stoplist = set(self.stoplist)

    def __iter__(self):
        for file_index in self.index:
            docs = json.load(open('/home/zengwei/DD/wxf_result/results/DD16-' + str(file_index) + '.json'))
            for doc in docs:
                contents = doc['content']
                content_tag = [doc['key']]
                # Lowercase each document, split it by white space and filter out stopwords
                content = [word for word in contents.lower().split() if word not in self.stoplist and word.isalpha()]
                yield TaggedDocument(content, content_tag)

            topics = json.load(open('/home/zengwei/DD/data/topic_info.json'))
            topic = [word for word in topics[file_index-1]['description'].lower().split() if word not in self.stoplist and word.isalpha()]
            topic_tag = [topics[file_index-1]['id']]
            yield TaggedDocument(topic, topic_tag)


documents = TaggedResultsDocuments('Ebola')

model = Doc2Vec(documents=documents, size=200, window=8, min_count=3,  workers=40)

for npochs in [10, 20, 50, 100]:

    model.train(documents, total_examples=model.corpus_count, epochs=npochs)
    print 'ndocs: ', len(model.docvecs.doctags)


    model.save('All_200_'+str(npochs)+'.model')
    model.save_word2vec_format('All_200_' + str(npochs) + '.d2v', doctag_vec=True, word_vec=False)

    resultfile = open('All_200_' + str(npochs) + '.yml', 'w')
    for tag in model.docvecs.doctags:
        info = yaml.dump({'tag': tag, 'feature': np.asarray(model.docvecs[tag], dtype='float64')})+'\n'
        resultfile.write(info)
