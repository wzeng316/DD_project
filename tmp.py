import tensorflow as tf
from Exp_gain import *
import datetime


def init_vec(dimension):
    return tf.Variable(tf.random_normal(dimension, stddev=0.01))


class DiverseRank(object):
    def __init__(self, lenepisode, nfeature, nhidden, lr):

        self.len_episode = lenepisode
        self.lr = lr
        self.nhidden = nhidden
        self.nfeature = nfeature

        self.ndoc = 1000
        self.ite=0

        global DD_learning_rate, DD_reward, DD_current_state, DD_query, DD_perceive_docs, DD_candidate_docs, DD_all_docs, DD_Ndoc

        global DD_query_state, DD_next_state, DD_score, DD_policy, DD_train_step, sess, DD_merge_train, DD_writer

        global DD_summary_reward, DD_summary_reward_soft, DD_summary_reward_train, DD_summary_reward_test, DD_merge_reward


        with tf.name_scope('input'):
            DD_learning_rate = tf.placeholder(tf.float32)
            DD_reward = tf.placeholder(tf.float32, None, name='reward')
            DD_current_state = tf.placeholder(tf.float32, [None, nhidden], name='current_state')
            DD_query = tf.placeholder(tf.float32, [None, nfeature], name='querys')
            DD_perceive_docs = tf.placeholder(tf.float32, [None, nfeature], name='docs')
            DD_candidate_docs = tf.placeholder(tf.float32, [None, nfeature], name='docs')
            DD_all_docs = tf.placeholder(tf.float32, [None, nfeature], name='candidate_docs')

            DD_summary_reward_soft = tf.placeholder(tf.float32, None, name='DD_summary_reward_soft')
            DD_summary_reward = tf.placeholder(tf.float32, None, name='DD_summary_reward')
            DD_summary_reward_train = tf.placeholder(tf.float32, None, name='DD_summary_reward_train')
            DD_summary_reward_test  = tf.placeholder(tf.float32, None, name='DD_summary_reward_test')


        with tf.variable_scope('model_query'):
            W_q = init_vec([nfeature, nhidden])
            DD_query_state = tf.tanh(tf.matmul(DD_query, W_q))

        with tf.variable_scope('rnn1'):
            cell = tf.contrib.rnn.GRUCell(nhidden)
            ep_split = tf.split(DD_perceive_docs, 1)
            _, DD_next_state = tf.contrib.rnn.static_rnn(cell, ep_split, initial_state=DD_current_state, dtype=tf.float32)
        W = init_vec([nfeature, nhidden])

        with tf.variable_scope('policy'):
            DD_score = tf.tanh(tf.matmul(DD_candidate_docs, tf.matmul(W, tf.transpose(DD_current_state))))
            DD_policy = tf.nn.softmax(tf.transpose(DD_score))

        with tf.variable_scope('rnn1', reuse=True) as scope:
            scope.reuse_variables()
            cell = tf.contrib.rnn.GRUCell(nhidden)

            ep_split_train = tf.split(DD_all_docs[0: self.len_episode], self.len_episode)
            DD_dynamic, _ = tf.contrib.rnn.static_rnn(cell, ep_split_train, initial_state=DD_query_state,
                                                      dtype=tf.float32)

            cross_entropy_list = []
            for position in range(self.len_episode):
                score = tf.tanh(
                    tf.matmul(DD_all_docs[self.len_episode+position:], tf.matmul(W, tf.transpose(DD_dynamic[position]))))
                policy = tf.nn.softmax(tf.transpose(score))
                cross_entropy_list.append(DD_reward[position] * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy,labels=[0]))
            cross_entropy=tf.add_n(cross_entropy_list)

        with tf.variable_scope('summary'):
            summary_train_reward = tf.summary.scalar('mean_reward', tf.reduce_mean(DD_reward))
            summary_train_max = tf.summary.scalar('max_reward', tf.reduce_max(DD_reward))
            summary_train_loss = tf.summary.scalar('cross_entropy', tf.reduce_sum(cross_entropy))

            summary_train_histogram = tf.summary.histogram('histogram', DD_reward)

        with tf.variable_scope('reward'):
            summary_reward_soft = tf.summary.scalar('reward_soft', tf.reduce_mean(DD_summary_reward_soft))
            summary_reward = tf.summary.scalar('reward', tf.reduce_mean(DD_summary_reward))
            summary_reward_train = tf.summary.scalar('reward_train', tf.reduce_mean(DD_summary_reward_train))
            summary_reward_test = tf.summary.scalar('reward_test',  tf.reduce_mean(DD_summary_reward_test))


        DD_train_step = tf.train.GradientDescentOptimizer(learning_rate=DD_learning_rate).minimize(cross_entropy)

        sess = tf.Session()

        DD_merge_train = tf.summary.merge([summary_train_reward, summary_train_max, summary_train_loss, summary_train_histogram])
        DD_merge_reward = tf.summary.merge([summary_reward, summary_reward_test, summary_reward_train, summary_reward_soft])

        DD_writer = tf.summary.FileWriter('/users/zengwei/DD/Tensorboard/'+ datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), sess.graph)

        init = tf.global_variables_initializer()

        sess.run(init)




    def gen_query_state(self, query):
        query_vec = np.asarray(query).reshape([1, self.nfeature])

        state = sess.run(DD_query_state, feed_dict={DD_query: query_vec})
        return state

    def gen_next_state(self, current_state, perceive_docs):
        current_state_vec = np.asarray(current_state).reshape([-1, self.nhidden])
        perceive_docs_vec = np.asarray(perceive_docs).reshape([-1, self.nfeature])

        state = sess.run(DD_next_state,
                         feed_dict={DD_current_state: current_state_vec, DD_perceive_docs: perceive_docs_vec})
        return state

    def gen_score(self, current_state, candidate_docs):
        current_state_vec = np.asarray(current_state).reshape([-1, self.nhidden])
        candidate_docs_vec = np.asarray(candidate_docs).reshape([-1, self.nfeature])

        score = sess.run(DD_score,
                         feed_dict={DD_current_state: current_state_vec, DD_candidate_docs: candidate_docs_vec})
        return score

    def gen_policy(self, current_state, candidate_docs):
        current_state_vec = np.asarray(current_state).reshape([-1, self.nhidden])
        candidate_docs_vec = np.asarray(candidate_docs).reshape([-1, self.nfeature])

        policy = sess.run(DD_policy,
                          feed_dict={DD_current_state: current_state_vec, DD_candidate_docs: candidate_docs_vec})
        return policy

    def gen_episode_softmax(self, queryvec, docvec, docs):
        current_state = self.gen_query_state(queryvec)

        ndoc = len(docvec)

        docd_id = docs[:]
        n_candidate = ndoc

        rank_list = []
        for ite in range(min(ndoc, self.len_episode)):
            policy = self.gen_policy(current_state, docvec)
            action = np.random.choice(n_candidate, 1, p=policy[0])[0]

            current_state = self.gen_next_state(current_state, docvec[action])

            n_candidate = n_candidate - 1
            rank_list.append(docd_id[action])
            queryvec.append(docvec[action])
            del docd_id[action]
            del docvec[action]

        return rank_list, queryvec + docvec

    def gen_episode_greedy(self, queryvec, docvec, docs):

        current_state = self.gen_query_state(queryvec)

        Ndoc = len(docvec)
        Nite = min(Ndoc, self.len_episode)

        doc_id = docs[:]

        rank_list = []
        for ite in range(Nite):
            score = self.gen_score(current_state, docvec)
            action = np.argmax(score)

            current_state = self.gen_next_state(current_state, docvec[action])

            rank_list.append(doc_id[action])
            queryvec.append(docvec[action])
            del doc_id[action]
            del docvec[action]

        return rank_list, queryvec + docvec

    def test_model(self, querydocs, doc2vec, truth):

        reward_list=[]
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

            reward = alphaDCG_reward_per_query(ranklist, truth[query])

            info = measure(ranklist, truth[query])

            # print reward[0],
            reward_list.append(reward[0])

        print '\n               ', sum(reward_list), '\n'

        return reward_list

    def train_model(self, querydocs, doc2vec, truth):
        querylist = querydocs.keys()
        train_data = {}

        reward_list=[]
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
            reward = alphaDCG_reward_per_query(ranklist, truth[query])
            reward_list.append(reward[0])
            # print 'reward', reward[0]
            train_data[query] = {'querydocvec': querydocvec, 'reward': reward}


        for query in querylist:
            querydocvec = train_data[query]['querydocvec']
            reward = train_data[query]['reward']
            self.optimize_model(querydocvec, reward)

        return reward_list

    def optimize_model(self, querydocvec, reward):
        ndoc = len(querydocvec)
        self.ndoc = len(querydocvec) - 1
        query_vec = np.asarray((querydocvec[0])).reshape([1, self.nfeature])
        doc_vec = np.asarray(querydocvec[1:ndoc + 1]).reshape([-1, self.nfeature])

        summary,_ = sess.run([DD_merge_train, DD_train_step], feed_dict={DD_query: query_vec, DD_all_docs: doc_vec, DD_reward: reward, DD_learning_rate: self.lr})
        DD_writer.add_summary(summary, self.ite)



    def main(self, nite, querydocs, doc2vec, truth):
        for n_iteration in range(nite):

            self.ite = n_iteration
            self.train_model(querydocs, doc2vec, truth)

            if self.ite % 10 == 0:
                self.test_model(querydocs, doc2vec, truth)


    def main_3K(self, nite, querydocs_3K, doc2vec, truth):

        for n_iteration in range(nite):

            self.ite = n_iteration
            reward_soft = self.train_model(dict(querydocs_3K[0], **querydocs_3K[1]), doc2vec, truth)

            if self.ite % 10 == 0:
                reward_train = self.test_model(querydocs_3K[2], doc2vec, truth)
                reward_test = self.test_model(dict(querydocs_3K[0], **querydocs_3K[1]), doc2vec, truth)

                summary= sess.run(DD_merge_reward, feed_dict={DD_summary_reward:np.asarray(reward_train+reward_test), DD_summary_reward_soft:np.asarray(reward_soft), DD_summary_reward_train:np.asarray(reward_train), DD_summary_reward_test:np.asarray(reward_test)})
                DD_writer.add_summary(summary, self.ite)

    def main_3KCV(self, nite, querydocs, doc2vec, truth):

        for n_iteration in range(nite):

            self.ite = n_iteration
            query_keys = querydocs.keys()

            query_index = np.arange(len(query_keys))
            np.random.shuffle(query_index)
            query_index = np.split(query_index, [36])

            train_data={}
            for i in query_index[0]:
                key = query_keys[i]
                train_data[key] = querydocs[key]
            test_data = {}
            for i in query_index[1]:
                key = query_keys[i]
                test_data[key] = querydocs[key]

            reward_soft =self.train_model(train_data, doc2vec, truth)

            if self.ite % 10 == 0:
                reward_train = self.test_model(test_data, doc2vec, truth)
                reward_test = self.test_model(train_data, doc2vec, truth)
                summary = sess.run(DD_merge_reward, feed_dict={DD_summary_reward: np.asarray(reward_train + reward_test),
                                                              DD_summary_reward_soft: np.asarray(reward_soft),
                                                              DD_summary_reward_train: np.asarray(reward_train),
                                                              DD_summary_reward_test: np.asarray(reward_test)})
                DD_writer.add_summary(summary, self.ite)

