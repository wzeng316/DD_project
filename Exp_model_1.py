import tensorflow as tf
from Exp_gain import *


def init_vec(dimension):
    return tf.Variable(tf.random_normal(dimension, stddev=0.01))


class DiverseRank(object):
    def __init__(self, lenepisode, nfeature, nhidden, lr):

        self.len_episode = lenepisode
        self.lr = lr
        self.top_n=10

        global learning_rate, query_docs, candidate_docs, score, policy, train_step, sess

        with tf.name_scope('input'):
            learning_rate = tf.placeholder(tf.float32)
            query_docs = tf.placeholder(tf.float32, [None, nfeature], name='docs')
            candidate_docs = tf.placeholder(tf.float32, [None, nfeature], name='candidate_docs')

        cell = tf.contrib.rnn.GRUCell(nhidden)

        with tf.name_scope('rnn'):
            ep_split = tf.split(query_docs, 1, 0, 'split')
            _, state = tf.contrib.rnn.static_rnn(cell, ep_split, dtype=tf.float32)

        with tf.name_scope('policy'):
            w = init_vec([nfeature, nhidden])
            score = tf.tanh(tf.matmul(candidate_docs, tf.matmul(w, tf.reshape(state[-1], [nhidden, 1]))))
            policy = tf.nn.softmax(tf.transpose(score))

        init = tf.global_variables_initializer()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy, labels=[0])

        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

        sess = tf.Session()
        sess.run(init)

    def gen_episode_softmax(self, queryvec, docvec, docs):
        ndoc = len(docvec)

        docd_id = docs[:]
        n_candidate = ndoc

        rank_list = []
        for ite in range(min(ndoc, self.len_episode)):
            prob = sess.run(policy,
                            feed_dict={query_docs: np.asanyarray(queryvec), candidate_docs: np.asanyarray(docvec)})
            action = np.random.choice(n_candidate, 1, p=prob[0])[0]

            n_candidate = n_candidate - 1
            rank_list.append(docd_id[action])
            queryvec.append(docvec[action])
            del docd_id[action]
            del docvec[action]

        return rank_list, queryvec + docvec

    def gen_episode_greedy(self, queryvec, docvec, docs):

        Ndoc = len(docvec)
        Nite = min(Ndoc, self.len_episode)

        doc_id = docs[:]

        rank_list = []
        for ite in range(Nite):
            scores = sess.run(score,
                              feed_dict={query_docs: np.asanyarray(queryvec), candidate_docs: np.asanyarray(docvec)})
            action = np.argmax(scores)

            rank_list.append(doc_id[action])
            queryvec.append(docvec[action])
            del doc_id[action]
            del docvec[action]

        return rank_list, queryvec + docvec

    def test_model(self, querydocs, doc2vec, truth):
        mean_alphaNDCG=np.zeros(self.top_n)
        nquery = len(querydocs.keys())
        for query in querydocs.keys():
            docs = querydocs[query][:]

            ndoc = len(docs)
            if ndoc < 2:
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', query
                continue

            querydocsvec = []
            querydocsvec.append(doc2vec[query])
            docsvec = []
            for doc in docs:
                docsvec.append(doc2vec[doc])

            # gen episode
            ranklist, _ = self.gen_episode_greedy(querydocsvec, docsvec, docs)

            the_alphaNDCG = alphaNDCG_per_query(ranklist, truth[query])
            mean_alphaNDCG += np.asarray(the_alphaNDCG)
            # print reward[0],
        mean_alphaNDCG = mean_alphaNDCG / nquery
        print '\n\t\t', mean_alphaNDCG.tolist()

    def train_model(self, querydocs, doc2vec, truth):
        querylist = querydocs.keys()
        train_data = {}

        mean_reward = 0

        nquery = len(querylist)
        for query in querylist:
            docs = querydocs[query][:]

            querydocsvec = []
            querydocsvec.append(doc2vec[query])
            docsvec = []
            for doc in docs:
                docsvec.append(doc2vec[doc])

            ranklist, querydocvec = self.gen_episode_softmax(querydocsvec, docsvec, docs)
            # print ranklist
            # print truth[query]
            reward = alphaDCG_gain_per_query(ranklist, truth[query])
            mean_reward += reward[0]
            # print 'reward', reward[0]
            train_data[query] = {'querydocvec': querydocvec, 'reward': reward}
        mean_reward = mean_reward/nquery
        print mean_reward,

        for query in querylist:
            querydocvec = train_data[query]['querydocvec']
            reward = train_data[query]['reward']
            self.optimize_model(querydocvec, reward)

    def optimize_model(self, querydocvec, reward):
        ndoc = len(querydocvec)
        for i in range(len(reward)):
            if reward[i] == 0:
                continue
            sess.run(train_step, feed_dict={query_docs: np.asanyarray(querydocvec[0:i + 1]), candidate_docs: np.asanyarray(querydocvec[i + 1:ndoc + 1]), learning_rate: self.lr * reward[i]})

    def main(self, nite, querydocs, doc2vec, query_document_topic):
        for ite in range(nite):
            self.train_model(querydocs, doc2vec, query_document_topic)

            if ite % 10 == 0:
                self.test_model(querydocs, doc2vec, query_document_topic)




class DDRank_1(object):
    def __init__(self, time, n_feature, n_hidden, lr):

        self.time = time
        self.group_size = 5
        self.lr = lr
        self.top_n = 10

        global learning_rate, query_docs, candidate_docs, score, policy, train_step, sess

        with tf.name_scope('input'):
            learning_rate = tf.placeholder(tf.float32)
            query_docs = tf.placeholder(tf.float32, [None, n_feature], name='docs')
            candidate_docs = tf.placeholder(tf.float32, [None, n_feature], name='candidate_docs')

        cell = tf.contrib.rnn.GRUCell(n_hidden)

        with tf.name_scope('rnn'):
            ep_split = tf.split(query_docs, 1, 0, 'split')
            _, state = tf.contrib.rnn.static_rnn(cell, ep_split, dtype=tf.float32)

        with tf.name_scope('policy'):
            w = init_vec([n_feature, n_hidden])
            score = tf.tanh(tf.matmul(candidate_docs, tf.matmul(w, tf.reshape(state[-1], [n_hidden, 1]))))
            policy = tf.nn.softmax(tf.transpose(score))

        init = tf.global_variables_initializer()
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy, labels=[0])

        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

        sess = tf.Session()
        sess.run(init)

    def gen_group_episode_softmax(self, query_vec, doc_vec, docs, truth):
        doc_id = docs[:]
        input = query_vec

        rank_list = []
        relevance_feedback = []
        for ite in range(self.time):
            relevance_feedback.append([])
            scores = sess.run(score, feed_dict={query_docs: np.asanyarray(input), candidate_docs: np.asanyarray(doc_vec)})
            scores = np.asarray(scores)[:,0]

            for i in range(self.group_size):
                prob = np.exp(scores)/np.sum(np.exp(scores))
                action = np.random.choice(len(prob), 1, p=prob)[0]

                scores = np.delete(scores, action)
                rank_list.append(doc_id[action])
                query_vec.append(doc_vec[action])
                if doc_id[action] in truth[0].keys():
                    relevance_feedback[ite].append(doc_vec[action])

                del doc_id[action]
                del doc_vec[action]

            # index = sorted(action_list)
            # for i in range(len(index)):
            #     del doc_id[index[i] - i]
            #     del doc_vec[index[i] - i]

            input = input + relevance_feedback[ite]

        return rank_list, query_vec + doc_vec, relevance_feedback

    def gen_group_episode_greedy(self, query_vec, doc_vec, docs, truth):
        doc_id = docs[:]

        rank_list = []
        for ite in range(self.time):
            scores = sess.run(score, feed_dict={query_docs: np.asanyarray(query_vec), candidate_docs: np.asanyarray(doc_vec)})

            for i in range(self.group_size):
                action = np.argmax(scores)

                scores = np.delete(scores, action)
                rank_list.append(doc_id[action])
                if doc_id[action] in truth[0].keys():
                    query_vec.append(doc_vec[action])
                del doc_id[action]
                del doc_vec[action]

        return rank_list

    def test_model(self, query_docs, doc2vec, truth):
        mean_alphaNDCG = np.zeros(self.top_n)
        n_query = len(query_docs.keys())

        for query in query_docs.keys():
            docs = query_docs[query][:]

            n_doc = len(docs)
            if n_doc < 2:
                print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', query
                continue

            query_docs_vec = []
            query_docs_vec.append(doc2vec[query])
            docs_vec = []
            for doc in docs:
                docs_vec.append(doc2vec[doc])

            # gen episode
            rank_list = self.gen_group_episode_greedy(query_docs_vec, docs_vec, docs, truth[query])

            the_alphaNDCG = alphaNDCG_per_query(rank_list, truth[query])
            mean_alphaNDCG += np.asarray(the_alphaNDCG)
            # print reward[0],
        mean_alphaNDCG = mean_alphaNDCG / n_query
        print '\n\t\t', mean_alphaNDCG.tolist()


    def train_model(self, querydocs, doc2vec, truth):
        querylist = querydocs.keys()
        train_data = {}

        mean_reward = 0
        nquery = len(querylist)
        for query in querylist:
            docs = querydocs[query][:]

            querydocsvec = []
            querydocsvec.append(doc2vec[query])
            docsvec = []
            for doc in docs:
                docsvec.append(doc2vec[doc])

            # print 'gen_episode_softmax'
            ranklist, querydocvec, feedback = self.gen_group_episode_softmax(querydocsvec, docsvec, docs, truth[query])
            # print ranklist
            reward = alphaDCG_group_reward_per_query(ranklist, truth[query])
            mean_reward += reward[0]
            # print 'reward', reward[0]
            train_data[query] = {'querydocvec': querydocvec, 'reward': reward, 'feedback': feedback}
        mean_reward = mean_reward / nquery
        print mean_reward,

        for query in querylist:
            querydocvec = train_data[query]['querydocvec']
            reward = train_data[query]['reward']
            feedback = train_data[query]['feedback']
            self.optimize_model(querydocvec, feedback, reward)

    def optimize_model(self, querydocvec, relevance_feedback, reward):
        ndoc = len(querydocvec)
        input = [querydocvec[0]]
        for time in range(len(reward)):
            if reward[time] == 0:
                continue
            for i in range(self.group_size):
                sess.run(train_step, feed_dict={query_docs: np.asanyarray(input), candidate_docs: np.asanyarray(querydocvec[time*self.group_size + i+1 :ndoc]), learning_rate: self.lr * reward[time]})

            input += relevance_feedback[time]

    def main(self, nite, querydocs, doc2vec, query_document_topic):
        for ite in range(nite):
            self.train_model(querydocs, doc2vec, query_document_topic)

            if ite % 10 == 0:
                self.test_model(querydocs, doc2vec, query_document_topic)


