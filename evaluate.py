import scipy
import utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import os
import sys
import pickle
import json
from scipy.special import softmax
import gensim
from gensim.models.coherencemodel import CoherenceModel

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def get_topics(topic_word, vocab, topn):
    topics = []
    for topic in topic_word:
        sorted_indices = np.argsort(topic)
        # topic_word_indices = sorted_indices[-topn:]
        topic_word_indices = np.flip(sorted_indices[-topn:])
        topics.append([vocab[idx] for idx in topic_word_indices])
    return topics

def topic_evaluation(doc_word, topic_word, vocab, n_list):
    res = {}
    # original evaluation
    res['c_npmi'] = utils.evaluate_NPMI(doc_word, topic_word)
    res['c_v'] = utils.evaluate_CV(doc_word, topic_word)
    res['TU'] = utils.evaluate_TU(topic_word)
    res['TQ'] = res['c_npmi'] * res['TU'] 

    
    # gensim evaluation
    # topics = get_topics(topic_word, vocab, np.max(np.array(n_list)))
    # texts = [" ".join([vocab[word_idx] for word_idx in doc]) for doc in doc_word]
    # for metric in ['c_v', 'c_npmi']:
    #     coh = .0
    #     for topn in n_list:
    #         cm = CoherenceModel(topics = topics, texts = texts, dictionary = vocab, coherence = metric, topn = topn)
    #         coh += cm.get_coherence()
    #     coh /= len(n_list)
    #     res['gensim_{}'.format(metric)] = coh
    return res

def doc_evaluation(train_doc_topic, test_doc_topic, train_labels_true, test_labels_true, n_clusters):
    # train_doc_topic = softmax(train_doc_topic, axis = 1)
    # test_doc_topic = softmax(test_doc_topic, axis = 1)
    res = {}
    n_topics = train_doc_topic.shape[1]
    #top Clustering
    # test_labels_pred = np.argmax(test_doc_topic, axis = 1)
    # res['top_ari'] = metrics.adjusted_rand_score(test_labels_true, test_labels_pred)
    # #res['top_fbeta'] = metrics.fbeta_score(test_labels_true, test_labels_pred, beta = 1)
    # res['top_nmi'] = metrics.normalized_mutual_info_score(test_labels_true, test_labels_pred)
    # res['top_purity'] = purity_score(test_labels_true, test_labels_pred)

    #KMeans Clustering
    kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 1234)
    kmeans.fit(train_doc_topic)
    test_labels_pred = kmeans.predict(test_doc_topic)

    res['kmeans_ari'] = metrics.adjusted_rand_score(test_labels_true, test_labels_pred)
    #res['kmeans_fbeta'] = metrics.fbeta_score(test_labels_true, test_labels_pred, beta = 1)
    res['kmeans_nmi'] = metrics.normalized_mutual_info_score(test_labels_true, test_labels_pred)
    res['kmeans_purity'] = purity_score(test_labels_true, test_labels_pred)

    # top-n clustering    
    # top_n = 5
    # top_n_indices = np.argpartition(train_doc_topic, -top_n)[-top_n:]    
    # mask = np.zeros_like(train_doc_topic, dtype = bool)
    # mask[top_n_indices] = True
    # new_train_doc_topic = train_doc_topic.copy()
    # new_train_doc_topic[~mask] = 0

    # top_n_indices = np.argpartition(test_doc_topic, -top_n)[-top_n:]    
    # mask = np.zeros_like(test_doc_topic, dtype = bool)
    # mask[top_n_indices] = True
    # new_test_doc_topic = test_doc_topic.copy()
    # new_test_doc_topic[~mask] = 0

    # kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 1234)
    # kmeans.fit(new_train_doc_topic)
    # test_labels_pred = kmeans.predict(new_test_doc_topic)

    # res['topn_ari'] = metrics.adjusted_rand_score(test_labels_true, test_labels_pred)
    # #res['kmeans_fbeta'] = metrics.fbeta_score(test_labels_true, test_labels_pred, beta = 1)
    # res['topn_nmi'] = metrics.normalized_mutual_info_score(test_labels_true, test_labels_pred)
    # res['topn_purity'] = purity_score(test_labels_true, test_labels_pred)

    #SVM Classification
    clf = svm.SVC()
    # clf = RandomForestClassifier()
    clf.fit(train_doc_topic, train_labels_true)
    test_labels_pred = clf.predict(test_doc_topic)
    res['acc'] = metrics.accuracy_score(test_labels_pred, test_labels_true)
    # res['macro_p'] = metrics.precision_score(test_labels_pred, test_labels_true, average = 'macro')
    # res['macro_r'] = metrics.recall_score(test_labels_pred, test_labels_true, average = 'macro')
    res['macro_f1'] = metrics.f1_score(test_labels_pred, test_labels_true, average = 'macro')
    # res['micro_f1'] = metrics.f1_score(test_labels_pred, test_labels_true, average = 'micro')
    # res['micro_p'] = metrics.precision_score(test_labels_pred, test_labels_true, average = 'micro')
    # res['micro_r'] = metrics.recall_score(test_labels_pred, test_labels_true, average = 'micro')
    
    return res

def load_data(source_dir, data_type = "train"):
    with open(os.path.join(source_dir, data_type + "_data.pkl"), "rb") as f:
        docs = pickle.load(f)
    with open(os.path.join(source_dir, data_type + "_label.pkl"), "rb") as f:
        labels = pickle.load(f)
    with open(os.path.join(source_dir, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    return docs, labels, vocab

def doc_word_mat(docs, vocab):
    vocab_size = len(vocab)
    doc_size = docs.shape[0]
    doc_word = np.zeros(shape = (doc_size, vocab_size), dtype = int)
    for i, doc in enumerate(docs):
        for word_idx in doc:
            doc_word[i][word_idx] += 1
    return doc_word

def load_res(model, dataset, exp_ini = None):
    source_dir = os.path.join("./data", dataset, "processed")
    _, train_labels, vocab = load_data(source_dir, "train")
    _, valid_labels, _ = load_data(source_dir, "valid")
    test_docs, test_labels, _ = load_data(source_dir, "test")
    doc_word = doc_word_mat(test_docs, vocab)
    train_labels = np.concatenate((train_labels, valid_labels), axis = 0)
    
    if model == 'ECRTM':
        res_dir = os.path.join("./baselines/ECRTM/ECRTM/output", dataset)
        data = scipy.io.loadmat(os.path.join(res_dir, "ECRTM_K100_1th_params.mat"))
        train_doc_topic = data['train_theta']
        test_doc_topic = data['test_theta']
        topic_word = data['beta']
        
    elif model in ['GINopic', 'GBTM', 'GNTM', 'ETM', 'ProdLDA', 'LDA']:
        res_dir = os.path.join("./baselines/GINopic/results", model, dataset)
        with open(os.path.join(res_dir, "info.pkl"), 'rb') as f:
            data = pickle.load(f)
        train_doc_topic = data['topic-document-matrix'].T.astype(np.float32)
        test_doc_topic = data['test-topic-document-matrix'].T.astype(np.float32)
        topic_word = data['topic-word-matrix']

        if model == 'GNTM':
            used_list = np.load(os.path.join('./baselines/GINopic/inter_res', model, dataset, "used_list.npy"))
            total_labels = np.concatenate((train_labels, test_labels), axis = 0)
            
            total_labels = total_labels[used_list]
            train_labels = total_labels[0: len(train_doc_topic)]
            test_labels = total_labels[len(train_doc_topic): ]
     
    elif model in ['ZeroShotTM', 'CombinedTM']:
        res_dir = os.path.join("./baselines/CTM/res", model , dataset)
        train_doc_topic = np.load(os.path.join(res_dir, "train_dt.npy"))
        test_doc_topic = np.load(os.path.join(res_dir, "test_dt.npy"))
        topic_word = np.load(os.path.join(res_dir, "tw.npy"))
        
    elif model == 'WeTe':
        res_dir = os.path.join("./baselines/", model, "res", dataset)
        with open(os.path.join(res_dir, "res.pkl"), 'rb') as f:
            data = pickle.load(f)
        train_doc_topic = data['train_theta']
        test_doc_topic = data['test_theta']
        topic_word = data['phi'].T
        
    elif model == 'CGTM':
        if exp_ini is None:
            res_dir = os.path.join("./cg_res", dataset)
        else:
            res_dir = os.path.join("./cg_res", exp_ini)
        topic_word = np.load(os.path.join(res_dir, "topic_word.npy"))
        train_doc_topic = np.load(os.path.join(res_dir, "train_theta.npy"))
        test_doc_topic = np.load(os.path.join(res_dir, "test_theta.npy"))

        # train_doc_topic = np.load(os.path.join(res_dir, "train_tf_theta.npy"))
        # test_doc_topic = np.load(os.path.join(res_dir, "test_tf_theta.npy"))
    
        # train_doc_topic = np.load(os.path.join(res_dir, "train_context_theta.npy"))
        # test_doc_topic = np.load(os.path.join(res_dir, "test_context_theta.npy"))

    elif model == 'NSTM':
        res_dir = os.path.join("./baselines", model, 'save', dataset)
        data = scipy.io.loadmat(os.path.join(res_dir, "save.mat"))
        topic_word = data['phi']
        train_doc_topic = data['train_theta']
        test_doc_topic = data['test_theta']

    elif model == 'BERTopic':
        res_dir = os.path.join('./baselines', model, 'res', dataset)
        with open(os.path.join(res_dir, 'topics.txt')) as f:
            topics = f.readlines()
        topics = [vocab.doc2bow(topic.split()) for topic in topics]
        topic_word = np.zeros((len(topics), len(vocab)), dtype = float)
        for i, topic in enumerate(topics):
            for (word_idx, count) in topic:
                topic_word[i][word_idx] = 1
        train_doc_topic = np.load(os.path.join(res_dir, 'train_dt.npy'))
        test_doc_topic = np.load(os.path.join(res_dir, 'test_dt.npy'))
        # topic_word = np.array(topics)
    
    elif model == 'FASTopic':
        res_dir = os.path.join('./baselines', model, 'res', dataset)
        topic_word = np.load(os.path.join(res_dir, 'tw.npy'))
        train_doc_topic = np.load(os.path.join(res_dir, 'train_dt.npy'))
        test_doc_topic = np.load(os.path.join(res_dir, 'test_dt.npy'))

    return doc_word, vocab, train_labels, test_labels, train_doc_topic, test_doc_topic, topic_word 
       
def save_res(model, dataset, res, file_name):
    save_dir = os.path.join("./res", model, dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, file_name), 'w') as f:
        f.write(json.dumps(res))
        
def load_embedding(dataset_name):
    source_dir = os.path.join("./data", dataset_name, "processed")
    # train_embedding = np.load(os.path.join(source_dir, "train_context.npz"))
    # valid_embedding = np.load(os.path.join(source_dir, "valid_context.npz"))
    # test_embedding = np.load(os.path.join(source_dir, "test_context.npz"))
    # train_embedding = scipy.sparse.csc_matrix((train_embedding['data'], train_embedding['indices'], train_embedding['indptr']), shape = train_embedding['shape']).toarray()
    # valid_embedding = scipy.sparse.csc_matrix((valid_embedding['data'], valid_embedding['indices'], valid_embedding['indptr']), shape = valid_embedding['shape']).toarray()
    # test_embedding = scipy.sparse.csc_matrix((test_embedding['data'], test_embedding['indices'], test_embedding['indptr']), shape = test_embedding['shape']).toarray()
    with open(os.path.join(source_dir, "train_contexts_data.pkl"), 'rb') as f:
        train_embedding = np.array(pickle.load(f))
    with open(os.path.join(source_dir, "valid_contexts_data.pkl"), 'rb') as f:
        valid_embedding = np.array(pickle.load(f))
    with open(os.path.join(source_dir, "test_contexts_data.pkl"), 'rb') as f:
        test_embedding = np.array(pickle.load(f))    
    train_embedding = np.concatenate((train_embedding, valid_embedding), axis = 0)
    return train_embedding, test_embedding

def test(dataset_name):
    train_embedding, test_embedding = load_embedding(dataset_name)
    source_dir = os.path.join("./data", dataset_name, "processed")
    _, train_labels, vocab = load_data(source_dir, "train")
    _, valid_labels, _ = load_data(source_dir, "valid")
    test_docs, test_labels, _ = load_data(source_dir, "test")
    # doc_word = doc_word_mat(test_docs, vocab)
    train_labels = np.concatenate((train_labels, valid_labels), axis = 0)
    n_labels = np.max(train_labels) + 1
    pca = PCA(n_components = 100)
    train_embedding = pca.fit_transform(train_embedding)
    test_embedding = pca.transform(test_embedding)
    res_doc = doc_evaluation(train_embedding, test_embedding, train_labels, test_labels, n_labels)
    print(res_doc)

if __name__ == "__main__":
    # model = sys.argv[1]
    # dataset = sys.argv[2]
    # doc_word, train_labels, test_labels, train_doc_topic, test_doc_topic, topic_word = load_res(model, dataset)
    # res_topic = topic_evaluation(doc_word, topic_word, 15)
    # res_doc = doc_evaluation(train_doc_topic, test_doc_topic, train_labels, test_labels, 100)
    # save_eval_res(model, dataset, res_topic, res_doc)
    
    model_list = ['FASTopic'] #['CGTM', 'FASTopic', 'GINopic', 'CombinedTM']
    dataset_list = ['20news', 'nyt', 'AGnews']
    for model in model_list:
        for dataset in dataset_list:
            print("Evaluating model: {} on dataset {}".format(model, dataset))
            n_list = [5, 10, 15]
            doc_word, vocab, train_labels, test_labels, train_doc_topic, test_doc_topic, topic_word = load_res(model, dataset)
            n_labels = len(np.unique(train_labels))
            print("n_labels: {}".format(n_labels))
            res_topic = topic_evaluation(doc_word, topic_word, vocab, n_list)
            print("Topic Result: ", res_topic, "\n")

            res_doc = doc_evaluation(train_doc_topic, test_doc_topic, train_labels, test_labels, n_labels)
            print("Document Result: ", res_doc, "\n")
            save_res(model, dataset, res_doc, "doc_res.json")

    # model = 'CGTM'
    # dataset = '20news'    
    # exp_list = ['20news_CI', '20news_GI', '20news_GIA', '20news_GMM', '20news_TA', '20news_TD', '20news_vMF']
    # n_list = [5, 10, 15]
    # for exp_ini in exp_list:
    #     print("Evaluating model: {} on setting {}".format(model, exp_ini))
    #     doc_word, vocab, train_labels, test_labels, train_doc_topic, test_doc_topic, topic_word = load_res(model, dataset, exp_ini)
    #     n_labels = len(np.unique(train_labels))
    #     print("n_labels: {}".format(n_labels))
    #     res_topic = topic_evaluation(doc_word, topic_word, vocab, n_list)
    #     print("Topic Result: ", res_topic, "\n")
    #     save_res(model, exp_ini, res_topic, "topic_res.json")
    #     res_doc = doc_evaluation(train_doc_topic, test_doc_topic, train_labels, test_labels, n_labels)
    #     print("Document Result: ", res_doc, "\n")
    #     save_res(model, exp_ini, res_doc, "doc_res.json")