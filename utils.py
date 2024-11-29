import numpy as np
#import matplotlib.pyplot as plt
import os
from gensim.models.coherencemodel import CoherenceModel
import requests
from sklearn.decomposition import PCA
from torch import nn
#import manifolds
import torch
from sklearn.neighbors import kneighbors_graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def compute_overlap(level1, level2):
    sum_overlap_score = 0.0
    for N in [5, 10, 15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        c = 0
        for n in word_idx1:
            if n in word_idx2:
                c += 1
        sum_overlap_score += c / N
    return sum_overlap_score / 3
def compute_hierarchical_affinity_new(topic_dist_1,topic_dist_2, relation):
    child_ha, unchild_ha = [], []
    topic_dist_1 = topic_dist_1 / np.linalg.norm(topic_dist_1, axis=1, keepdims=True)
    topic_dist_2 = topic_dist_2 / np.linalg.norm(topic_dist_2, axis=1, keepdims=True)
    for index in range(32):
        for index_2 in range(128):
            ha = topic_dist_1[index].dot(topic_dist_2[index_2])   
            if (index,index_2) in relation:
                child_ha.append(ha)
            else:
                unchild_ha.append(ha)  
    return np.mean(child_ha), np.mean(unchild_ha)  
def compute_hierarchical_affinity_new_2(topic_dist_1,topic_dist_2, relation):
    child_ha, unchild_ha = [], []
    topic_dist_1 = topic_dist_1 / np.linalg.norm(topic_dist_1, axis=1, keepdims=True)
    topic_dist_2 = topic_dist_2 / np.linalg.norm(topic_dist_2, axis=1, keepdims=True)
    for index in range(8):
        for index_2 in range(32):
            ha = topic_dist_1[index].dot(topic_dist_2[index_2])   
            if (index,index_2) in relation:
                child_ha.append(ha)
            else: 
                unchild_ha.append(ha)  
    return np.mean(child_ha), np.mean(unchild_ha)  
def compute_hierarchical_affinity(topic_dist, relation):
    child_ha, unchild_ha = [], []
    topic_dist = topic_dist / np.linalg.norm(topic_dist, axis=1, keepdims=True)

    for child_index,child in enumerate(relation[0]):
        for parent in relation[1]:
            if parent == child:
                continue
            ha = topic_dist[child] * topic_dist[parent]
            if relation[1][child_index] == parent:
                child_ha.append(ha)
            else:
                unchild_ha.append(ha)           
    return np.mean(child_ha), np.mean(unchild_ha)
def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    total = sum(topic_words,[])
    return len(vocab) / len(total)

def build_bert_embedding(embedding_fn, vocab, data_dir):
    print(f"building bert embedding matrix for dit {len(vocab)}")
    tokenize = BertTokenizer.from_pretrained('bert-base-uncased')
    embedding_mat_fn = os.path.join(data_dir, f"bert_emb_{len(vocab)}.npy")

    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat

    # build bert mat
    index = np.array(
        tokenize.encode(list(vocab.token2id.keys()), add_special_tokens=False))
    bert_mat = np.load(embedding_fn)
    bert_emb = bert_mat[index]
    np.save(embedding_mat_fn, bert_emb)
    return bert_emb
# def compute_topic_specialization(topic_word, corpus_topic):
#     print('topic_word',topic_word.shape,corpus_topic.shape)
#     topics_vec = topic_word.sum(axis=0)
#     corpus_topic=corpus_topic.sum(axis=0)
#     print(topics_vec.shape)
#     topics_vec = topics_vec / np.linalg.norm(topics_vec)
#     corpus_topic=corpus_topic/np.linalg.norm(corpus_topic)
#     topics_spec = 1 - topics_vec.dot(corpus_topic.T)
#
#     return topics_spec

def compute_topic_specialization(topic_word, corpus_topic):
  #  print('topic_word',topic_word.shape,corpus_topic.shape)
    if topic_word.shape[0] > 0:
      for i in range(topic_word.shape[0]):
                topic_word[i] = topic_word[i] / np.linalg.norm(topic_word[i])
    corpus_topic=corpus_topic.sum(axis=0)
    topics_vec=topic_word
    corpus_topic=corpus_topic/np.linalg.norm(corpus_topic)
    topics_spec = 1 - topics_vec.dot(corpus_topic.T)
    topics_spec=np.mean(topics_spec)
    return topics_spec
# def compute_topic_specialization(topic_word, corpus_topic):
#     print('topic_word',topic_word.shape,corpus_topic.shape)
#     topics_vec = topic_word
#     if topics_vec.shape[0] > 0:
#         for i in range(topics_vec.shape[0]):
#             topics_vec[i] = topics_vec[i] / np.linalg.norm(topics_vec[i])
#         for i in range(corpus_topic.shape[0]):
#             corpus_topic[i] = corpus_topic[i] / np.linalg.norm(corpus_topic[i])
#         topics_spec = 1 - topics_vec.dot(corpus_topic.T)
#         #print(topics_vec.dot(corpus_topic.T))
#        # print(topics_spec.shape)
#         depth_spec = np.mean(topics_spec)
#         return depth_spec
#     else:
#         return 0
def evaluate_NPMI(test_data, topic_dist, n_list=[5,10,15]):
    NPMI = 0.0
    for n in n_list:
        NPMI += compute_coherence(test_data, topic_dist, n)
    NPMI /= len(n_list)
    return NPMI
def evaluate_CV(test_data, topic_dist, n_list=[5,10,15]):
    CV = 0.0
    for n in n_list:
        CV += compute_cv(test_data, topic_dist, n)
    CV /= len(n_list)
    return CV
def evaluate_WE(test_data, topic_dist,pre_embedding,n_list=[5,10,15]):
    WE = 0.0
    for n in n_list:
        WE += compute_We(test_data, topic_dist, n,pre_embedding)
    WE /= len(n_list)
    return WE

def evaluate_NPMI2(test_data, topic_dist, n_list=[5,10,15]):
    coh_list_n=[]
    for n in n_list:
        coh_list=compute_coherence2(test_data, topic_dist, n)
        coh_list_n= np.concatenate((coh_list_n,coh_list),axis=0)
    return coh_list_n
    
def evaluate_TU(topic_word, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_TU(topic_word, n)
    TU /= len(n_list)
    return TU

def evaluate_TD(topic_word, n_list=[5,10,15]):
    TU = 0.0
    for n in n_list:
        TU += compute_TD(topic_word, n)
    TU /= len(n_list)
    return TU

def compute_TD(topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    if topic_size == 0:
        return 0
    else:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TD = len(np.unique(np.array(topic_list).flatten()))/len(np.array(topic_list).flatten())
        return TD

def compute_TU(topic_word, N):
    topic_size, word_size = np.shape(topic_word)
    if topic_size == 0:
        return 0
    else:
        topic_list = []
        for topic_idx in range(topic_size):
            top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
            topic_list.append(top_word_idx)
        TU = 0
        cnt = [0 for i in range(word_size)]
        for topic in topic_list:
            for word in topic:
                cnt[word] += 1
        for topic in topic_list:
            TU_t = 0
            for word in topic:
                TU_t += 1 / cnt[word]
            TU_t /= N
            TU += TU_t
        TU /= topic_size
        return TU
def build_level(adj_matrix,flag = 0):
    adj_matrix= adj_matrix.T.detach().to("cpu").numpy()

    adj_matrix_new = np.array(adj_matrix.copy())
    for index in range(len(adj_matrix)):
        adj_matrix_new[index][adj_matrix_new[index]==np.max(adj_matrix_new[index])] = -1
        adj_matrix_new[index][adj_matrix_new[index]==np.max(adj_matrix_new[index])] = -1
    adj_matrix_new[adj_matrix_new != -1] = 0
    adj_matrix_new[adj_matrix_new == -1] = 1
    #print('trees',adj_matrix_new.shape,adj_matrix_new)
    trees = adj_matrix_new
    
    relation = np.where(adj_matrix_new == 1)
    print('len(relation[0])',len(relation[0]))
    relation = list(zip(relation[0], relation[1]))
    return trees, relation

def compute_clnpmi(level1, level2, doc_word):

    sum_coherence_score = 0.0
    c = 0

    for N in [5,10,15]:
        word_idx1 = np.argpartition(level1, -N)[-N:]
        word_idx2 = np.argpartition(level2, -N)[-N:]
        
        sum_score = 0.0
        set1 = set(word_idx1)
        set2 = set(word_idx2)
        inter = set1.intersection(set2)
        word_idx1 = list(set1.difference(inter))
        word_idx2 = list(set2.difference(inter))

        for n in range(len(word_idx1)):
            flag_n = doc_word[:, word_idx1[n]] > 0
            p_n = np.sum(flag_n) / len(doc_word)
            for l in range(len(word_idx2)):
                flag_l = doc_word[:, word_idx2[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_nl == len(doc_word):
                    sum_score += 1
                elif p_n * p_l * p_nl > 0:
                    p_l = p_l / len(doc_word)
                    p_nl = p_nl / len(doc_word)
                    p_nl += 1e-10
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                c += 1
        if c > 0:
            sum_score /= c
        else:
            sum_score = 0
        sum_coherence_score += sum_score
    return sum_coherence_score / 3
def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    embedding_mat_fn = os.path.join(data_dir, f"embedding_mat_{len(vocab)}.npy")
    
    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    
    # build embedding mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
            
    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros((len(vocab) + 1, embedding_dim))    # -1 is for padding
    for i,word  in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat
    
def build_embedding(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix for dict {len(vocab)} if need...")
    embedding_mat_fn = os.path.join(data_dir, f"embedding_mat_{len(vocab)}.npy")
    
    # if matrix exists
    if os.path.exists(embedding_mat_fn):
        embedding_mat = np.load(embedding_mat_fn)
        return embedding_mat
    
    # build embedding mat
    embedding_index = {}
    with open(embedding_fn, encoding='UTF-8') as fin:
        first_line = True
        l_id = 0
        for line in fin:
            if l_id % 100000 == 0:
                print("loaded %d words embedding..." % l_id)
            if ("glove" not in embedding_fn) and first_line:
                first_line = False
                continue
            line = line.rstrip()
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
            l_id += 1
            
    embedding_dim = len(list(embedding_index.values())[0])
    embedding_mat = np.zeros((len(vocab) + 1, embedding_dim))    # -1 is for padding
    for i,word  in vocab.items():
        embedding_vec = embedding_index.get(word)
        if embedding_vec is not None:
            embedding_mat[i] = embedding_vec
    np.save(embedding_mat_fn, embedding_mat)
    return embedding_mat

def build_embedding_PCA(embedding_fn, vocab, data_dir):
    print(f"building embedding matrix to PCA for dict {len(vocab)} if need...")
    embedding_PCA_fn = os.path.join(data_dir, f"embedding_mat_PCA_{len(vocab)}.npy")

    if os.path.exists(embedding_PCA_fn):
        embedding_PCA_mat = np.load(embedding_PCA_fn)
        return embedding_PCA_mat
    
    embedding_mat = build_embedding(embedding_fn, vocab, data_dir)
    pca = PCA(n_components=2)
    embedding_PCA_mat = pca.fit_transform(embedding_mat)
    np.save(embedding_PCA_fn, embedding_PCA_mat)
    return embedding_PCA_mat

def compute_coherence(doc_word, topic_word, N):
    # print('computing coherence ...')    
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        sum_coherence_score += sum_score * (2 / (N * N - N))
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score

def compute_coherence2(doc_word, topic_word, N):
    # print('computing coherence ...')
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)

    # compute coherence of each topic
    coh_list = []
    for i in range(topic_size):
        word_array = topic_list[i]
        sum_score = 0.0
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    sum_score += np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
        coh_list.append( sum_score * (2 / (N * N - N)))
 #   print('coh_list',len(coh_list))
   # sum_coherence_score = coh_list / topic_size
    return coh_list

def evaluate_coherence(topic_words, texts, vocab):
    coherence = {}
    methods = ["c_v", "c_npmi", "c_uci", "u_mass"]
    for method in methods:
        coherence[method] = CoherenceModel(topics=topic_words, texts=texts, dictionary=vocab, coherence=method).get_coherence()
    return coherence
    

def evaluate_topic_diversity(topic_words):
    '''topic_words is in the form of [[w11,w12,...],[w21,w22,...]]'''
    vocab = set(sum(topic_words,[]))
    total = sum(topic_words,[])
    return len(vocab) / len(total)


def print_topic_word(topic_word, vocab, N):
    topic_size, word_size = np.shape(topic_word)

    top_word_idx = np.argsort(topic_word, axis=1)
    top_word_N = top_word_idx[:,-N:]

    for k, top_word_k in enumerate(top_word_N[:,::-1]):
        top_words = [vocab[id] for id in top_word_k]
        print(f'Topic {k}:{top_words}')


def get_palmetto(topic, url):
    res = requests.get(url, {'words' : ' '.join(topic)})
    coh = np.float(res.text)
    return coh



def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.show()

def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc="best")

def Co_Graph(reader,data_type,adj_alpha,device):
    (
        train_data2,
        train_label2,
        train_text2,
        _,
    ) = reader.get_sparse_matrix(data_type, mode="tfidf")  # 可以从reader内修改mode，改为tfidf
    ALL_data2 = train_data2.toarray()
    ALL_data2 = torch.from_numpy(ALL_data2).to(device)
    y=ALL_data2
    zero=torch.zeros_like(y)
    y = torch.where(y <adj_alpha, zero, y)
    one=torch.ones_like(y)
    y = torch.where(y >=adj_alpha, one, y)  #对于邻接矩阵的构造
    D = torch.sum(y.t(), dim=1, keepdim=True)
    A = torch.matmul(y.t(), y)  # 邻接矩阵 3531*N
    A_hat = A+ torch.eye(A.size(0), dtype=A.dtype,device=device)
    d=torch.pow(D+1, -1 / 2)
    d = torch.diag_embed(d.squeeze())
    lap_matrix=d@A_hat@d
    return lap_matrix






def convert_to_tensor(data,device):
 # 检查data是否为NumPy数组
 if isinstance(data, np.ndarray):
# 如果是NumPy数组，则转换为PyTorch张量
  return torch.from_numpy(data).to(device).float()
 elif isinstance(data, torch.Tensor):
# 如果已经是PyTorch张量，则直接返回
  return data.to(device).float()
 else:
# 如果既不是NumPy数组也不是PyTorch张量，则抛出异常或进行其他处理
  raise ValueError("Unsupported data type. Expected np.ndarray or torch.Tensor.")
class Reconst_Graph(nn.Module):
    def __init__(self):
        super(Reconst_Graph, self).__init__()

    def forward(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.

        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
       # print(adj_pred)
       # print(adj)
        loss = torch.mean(torch.pow(adj - adj_pred, 2))

        return loss,adj_pred


def matrix_cos(X):
        N=X.shape[0]
        #计算余弦相似度
       # print(X.shape)
        cosine_similarities = cosine_similarity(X)
        # 获取上三角部分的行和列索引
        row_idx, col_idx = np.triu_indices(N, k=1)  # k=1表示排除主对角线
        # 使用索引来选择元素并求和
        sum_upper_triangle = np.sum(cosine_similarities[row_idx, col_idx])/(N*(N-1)/2)
        return  sum_upper_triangle

def compute_cv(doc_word, topic_word, N):
    # print('computing coherence ...')
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        vector_matrix = np.eye(N)
        for n in range(N):
            flag_n = doc_word[:, word_array[n]] > 0
            p_n = np.sum(flag_n) / doc_size
            for l in range(n + 1, N):
                flag_l = doc_word[:, word_array[l]] > 0
                p_l = np.sum(flag_l)
                p_nl = np.sum(flag_n * flag_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    npmi= np.log(p_nl / (p_l * p_n)) / -np.log(p_nl)
                    vector_matrix[n][l]=npmi
                    vector_matrix[l][n]=npmi
        c_v=matrix_cos(vector_matrix)
        sum_coherence_score+=c_v
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score

def compute_We(doc_word, topic_word, N,WE):
    # print('computing coherence ...')
    #print(WE.shape)  #3531 300
    topic_size, word_size = np.shape(topic_word)
   # print('topic_size',topic_size,word_size)
    doc_size = np.shape(doc_word)[0]
    # find top words'index of each topic
    topic_list = []
    for topic_idx in range(topic_size):
        top_word_idx = np.argpartition(topic_word[topic_idx, :], -N)[-N:]
        topic_list.append(top_word_idx)
    # compute coherence of each topic
    sum_coherence_score = 0.0
    for i in range(topic_size):
        word_array = topic_list[i]
        vector_matrix = np.zeros((N, WE.shape[1]))
        for n in range(N):
           # print(word_array[n])
            vector_matrix[n]=WE[word_array[n]]
        #print('vector_matrix',vector_matrix)
        c_v=matrix_cos(vector_matrix)
        sum_coherence_score+=c_v
    sum_coherence_score = sum_coherence_score / topic_size
    return sum_coherence_score
def mask_func(adj, epsilon=0, mask_value=-1e16):
    mask = (adj > epsilon).detach().float()
    update_adj = adj * mask + (1 - mask) * mask_value
    return update_adj
class Reconst_Graph(nn.Module):
    def __init__(self):
        super(Reconst_Graph, self).__init__()

    def forward(self, emb, adj):
        '''
        Parameters
        ----------
        emb : Tensor
            An MxE tensor, the embedding of the ith node is stored in emb[i,:].
        adj : Tensor
            An MxM tensor, adjacent matrix of the graph.

        Returns
        -------
        loss : float
            The link prediction loss.
        '''
        emb_norm = emb.norm(dim=1, keepdim=True)
        emb_norm = emb / (emb_norm + 1e-6)
        adj_pred = torch.matmul(emb_norm, emb_norm.t())
       # print(adj_pred)
       # print(adj)

        loss = torch.mean(torch.pow(adj - adj_pred, 2))

        return loss,adj_pred