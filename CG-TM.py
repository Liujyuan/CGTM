from audioop import bias
import math
import os
from pickle import FALSE, TRUE
from re import X
#import hypertools as hyp
import evaluate
import time
import scipy.sparse as sp
from utils import *
from learning_utils import *
from scipy import sparse
import numpy as np
from pyparsing import Word
import torch.optim as optim
import yaml
from numpy.random import normal
inv_flag = False
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import init
from tqdm import tqdm
import utils
from GIA_layer import *
from reader_cont import TextReader
#from get_context_repr import *
import sys

#from sentence_transformers import SentenceTransformer
Tensor = torch.cuda.FloatTensor
np.random.seed(0)
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(inv_flag)
plot_map=False

def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class LossFunctions:
    eps = 1e-8
    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type="mse"):
        epsilon = 1e-8
      #  print('min',torch.min(predicted))
    #    print("Data type:", predicted.dtype)
        predicted.clamp_(min=epsilon)
     #   print(torch.min(predicted))
        if rec_type == "mse":
            if dropout_mask is None:
             #   loss = -torch.sum(torch.log(predicted+epsilon) * (real))
             loss = -torch.sum(torch.log(predicted) * (real))
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )
        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none").mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var):

        if self.eps > 0.0:
            var = var + self.eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).to(device)).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_loss(
        self, z, z_mu, z_var, z_mu_prior, z_var_prior
    ):  
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.sum(targets * log_q, dim=-1))

class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)
        self.act=nn.ReLU()
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)

        return mu, logvar

class Gaussian2(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian2, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)
        self.act=nn.Tanh()
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)

        mu=self.act(mu)
        logvar=self.act(logvar)
        return mu, logvar

# Encoder
class InferenceNet(nn.Module):
    def __init__(self,topic_num, x_dim, z_dim, y_dim, hidden_dim, nonLinear):
        super(InferenceNet, self).__init__()
       # self.encoder = nn.Sequential(nn.Linear(topic_num,topic_num), nn.BatchNorm1d(topic_num), nonLinear)

        # self.inference_qyx3 = torch.nn.ModuleList(
        #     [
        #         nn.Linear(topic_num, 300),  # 64 1
        #         nn.BatchNorm1d(300),
        #         nonLinear,
        #         GumbelSoftmax(300, y_dim),  # 1 256
        #     ]
        # )
        self.inference_qzyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num, 300),
                nn.BatchNorm1d(300),
                nonLinear,
                Gaussian(300, topic_num),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    # def qyx3(self, x,temperature,hard):
    #     num_layers = len(self.inference_qyx3)
    #     for i, layer in enumerate(self.inference_qyx3):
    #         if i == num_layers - 1:
    #             x = layer(x, temperature, hard)
    #         else:
    #             x = layer(x)
    #     return x
    def qzxy3(self, x):
       # concat = torch.cat((x.squeeze(2), y), dim=1)
        for i, layer in enumerate(self.inference_qzyx3):
          #  print(x.shape)
            x = layer(x)
            if i==2:
               theta=x
               #print('i',theta)
        return x,theta


    def forward(self, x,proj_theta, temperature, hard = 0):

           # print('x',x.shape)
           #   x = x.squeeze(2)
        # logits_3, prob_3, y_3  = self.qyx3(proj_theta,temperature, hard = 0)
        [mu_3, logvar_3],theta = self.qzxy3(x)
        var_3 = torch.exp(logvar_3)
        z_3 = self.reparameterize(mu_3, var_3)
        output_3 = {"mean": mu_3, "var": logvar_3, "gaussian": z_3}
        return output_3   ,theta

# Decoder
class GenerativeNet(nn.Module):
    def __init__(self,topic_num, x_dim=1, z_dim=1, y_dim=10, nonLinear=None):
        super(GenerativeNet, self).__init__()
        self.y_mu_1 = nn.Sequential(nn.Linear(y_dim, topic_num))
        self.y_var_1 = nn.Sequential(nn.Linear(y_dim, topic_num))

        if True:
            print('Constraining decoder to positive weights', flush=True)


        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num),
                nonLinear,
            ]
        )

    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z

    def forward(
        self,
        z,
    ):
        out = self.pxz(z)

        output = {"x_rec": out}
        return output

class net(nn.Module):
    def __init__(
        self,
        max_topic_num=64,
        batch_size=None,
        mask=None,
        emb_mat=None,
        topic_num=None,
        vocab_num=None,
        hidden_num=None,
        prior_beta=None,
            Adj_Graph=None,
            epsilon_g=None,
            gamma_c=None,
            gamma_g=None,
            GMM_weight=None,
        **kwargs,
    ):
        super(net, self).__init__()
        print("net topic_num={}".format(topic_num))

        self.gamma_c = gamma_c
        self.gamma_g = gamma_g
        self.GMM_weight=GMM_weight
        self.epsilon_g=epsilon_g
        self.dropout = nn.Dropout(0.1)
        self.max_topic_num = max_topic_num
        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        if emb_mat == None:
            self.word_embed = nn.Parameter(torch.rand(hidden_num, vocab_num))
        else:
            print("Using pre-train word embedding")
            self.word_embed = nn.Parameter(emb_mat)

        self.topic_embed = nn.Parameter(xavier_init.sample((topic_num, hidden_num)))
        self.GIA_emb=torch.zeros(hidden_num, vocab_num)
        self.GIA=GIA(vocab_num,hidden_num,[epsilon_g])  #0.004 0.4

        self.bert_proj2 = nn.Sequential(nn.Linear(vocab_num, max_topic_num), nn.BatchNorm1d(max_topic_num),
                                        nn.Tanh())
        self.encoder = nn.Sequential(nn.Linear(vocab_num, max_topic_num), nn.BatchNorm1d(max_topic_num), nn.Tanh())
        x_dim, y_dim, z_dim = 64, 10, 10  # x:  y:   z:
        hidden_num_x=300
        self.inference = InferenceNet( topic_num,x_dim, y_dim,z_dim, hidden_num_x, nn.Tanh())
        self.generative = GenerativeNet(topic_num,x_dim, y_dim,z_dim, nn.Tanh())
        self.losses = LossFunctions()
        self.Adj_Graph=Adj_Graph
        self.Reconst_Graph=Reconst_Graph()
        self.loss_cpt = nn.CrossEntropyLoss()
        self.bert_proj = nn.Sequential(
            nn.Linear(768, vocab_num), #hidden_num
)
        self.Topic_wise_Fusion=nn.Sequential(
            nn.Linear(max_topic_num+max_topic_num, max_topic_num), #hidden_num
              nn.ReLU(),

)

        for m in self.modules():
            if (
                type(m) == nn.Linear
                or type(m) == nn.Conv2d
                or type(m) == nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

        self.a = 1 * np.ones((1, self.max_topic_num)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / self.max_topic_num))).T + (
                    1.0 / (self.max_topic_num * self.max_topic_num)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False
    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        # KLD: N*K
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.max_topic_num)
        KLD = KLD.mean()
        return KLD

    def to_np(self,x):
        return x.cpu().detach().numpy()

    def get_topic_dist(self,):
        #    return torch.softmax(self.topic_embed @ self.gcn_emb.to(device), dim=1)
        gcn_emb=self.GIA(self.word_embed, self.Adj_Graph[0])
        return torch.softmax(self.topic_embed @ gcn_emb.to(device), dim=1)

    def encode(self, x):
        p1 = self.encoder(x)
        return p1

    def decode(self, x_ori, out,detach):
        out = torch.softmax(out, dim=1)
        self.topic_embed.data = F.normalize(self.topic_embed.data, dim=-1)  # 8,300
        word_embed = self.word_embed.detach() if detach else self.word_embed
        beta = torch.softmax( self.topic_embed @word_embed, dim=1)
        p = out @ beta  # #
        return p.T


    def get_sharpened(self, preds):
        targets = preds ** 2 / preds.sum(dim=0)
        #  print('targets',targets.shape)
        #  print('targets.sum(dim=1)',targets.sum(dim=1))
        targets = (targets.t() / targets.sum(dim=1)).t()

        return targets

    def TS_path(self, theta):
        # print("Theta shape: ", theta.shape)#B,K
        s_theta = torch.softmax(theta, dim=1)
        targets = self.get_sharpened(s_theta).detach()
        clus_loss = F.kl_div(s_theta.log(), targets, reduction='sum')

        Cluster_loss = clus_loss
        return Cluster_loss

    def get_fai(self):
        beta = torch.softmax(self.topic_embed @ self.word_embed, dim=-1)
        fai = beta.T
        return fai

    def get_theta(self, x, x_context):
        proj_theta = self.bert_proj(x_context)  # proj x_bert [B,bert_dim] to [B.hidden_dim]
        x_ori = x
        x_encode = self.encode(x)

        #about TF-IDF Encoder
        out_inf_x, theta = self.inference(
            x_encode, x_encode, 1, x_ori.view(x.size(0), -1, 1)
        )
        z = out_inf_x["gaussian"]
        output = self.generative(  # here
            z,
        )  # here

        dec_tf = output["x_rec"]

        # about Contextual Encoder
        proj_theta2 = self.bert_proj2(proj_theta)
        out_inf_1, theta = self.inference(  # x_new / x_hidden /x_encode
            proj_theta2, proj_theta2, 1, x_ori.view(x.size(0), -1, 1)
        )
        z = out_inf_1["gaussian"]
        output = self.generative(  # here
            z,
        )  # here


        dec_con = output["x_rec"]
        dec_cat = self.Topic_wise_Fusion(torch.cat((dec_tf, dec_con), -1))

        return dec_tf, dec_con, dec_cat


    def forward(self, x,x_context, dropout_mask=None, temperature=1.0, hard=0):
        self.GIA_emb = self.GIA(self.word_embed, self.Adj_Graph[0])
        proj_theta = self.bert_proj(x_context)
        x_ori = x
        x_encode = self.encode(x)

        #TF-IDF Encoder
        out_inf_x, theta = self.inference(
            x_encode, x_encode, temperature, x_ori.view(x.size(0), -1, 1)
        )
        z = out_inf_x["gaussian"]
        output= self.generative(  # here
            z,
        )  # here
        Theta_B= output["x_rec"]
        x_B = self.decode(x_ori,Theta_B,False)

        loss_rec_B = self.losses.reconstruction_loss(
            x_ori, x_B.T, dropout_mask, "mse"
        )

        #Contextual Encoder
        proj_theta2 = self.bert_proj2(proj_theta)
        out_inf_1, theta = self.inference(  # x_new / x_hidden /x_encode
            proj_theta2, proj_theta2, temperature, x_ori.view(x.size(0), -1, 1)
        )
        z = out_inf_1["gaussian"]
        output = self.generative(  # here
            z,
        )  # here
        Theta_C = output["x_rec"]
        x_C = self.decode(x_ori, Theta_C,False)
        loss_rec_C = self.losses.reconstruction_loss(
            x_ori, x_C.T, dropout_mask, "mse"
        )


        Theta_GMM = self.Topic_wise_Fusion(torch.cat((Theta_B, Theta_C), -1))

        dec_res = self.decode(x_ori, Theta_GMM, False)
        loss_rec_GMM = self.losses.reconstruction_loss(
            x_ori, dec_res.T, dropout_mask, "mse"
        )

        _,rec_graph= self.Reconst_Graph(self.GIA_emb.T, self.Adj_Graph[0]) #
        Topic_Sharpening_loss = self.TS_path(Theta_GMM)
        gs_kl_loss_B=self.compute_loss_KL(      out_inf_x["mean"],out_inf_x["var"])
        gs_kl_loss_C = self.compute_loss_KL(out_inf_1["mean"], out_inf_1["var"])
        GR_kl_loss = torch.nn.KLDivLoss(reduction="none")

        GR_loss = GR_kl_loss(torch.log(torch.softmax(mask_func(rec_graph), dim=-1) + 1e-9),
                             torch.softmax(mask_func(self.Adj_Graph[0]), dim=-1)).sum()


        loss_elbo = gs_kl_loss_B + gs_kl_loss_C + loss_rec_GMM
        loss_TA = loss_rec_B + loss_rec_C
        loss_TS = Topic_Sharpening_loss
        loss_GI = GR_loss
        loss_CI =  loss_TA  + self.GMM_weight*loss_TS  #GMM_weight =lamda in the paper

        loss = loss_elbo + self.gamma_c * loss_CI + self.gamma_g * loss_GI  #
        loss = loss / 2

        return loss

class AMM_no_dag(object):
    def __init__(
        self,
        reader=None,
        max_topic_num=64,
        model_path=None,
        emb_mat=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
            epsilon_g=None,
            gamma_c=None,
            gamma_g=None,
            GMM_weight=None,
            Adj_Graph=None,



        **kwargs,
    ):
        # prepare dataset
        if reader == None:
            raise Exception(" [!] Expected data reader")

        self.reader = reader
        self.model_path = model_path
        self.n_classes = self.reader.get_n_classes()  # document class
        self.topic_num_1 = topic_num_1
        self.topic_num_2 = topic_num_2
        self.topic_num_3 = topic_num_3
        self.pre_embeddings=emb_mat
        self.adj = self.initalize_A(topic_num_3)  # topic_num_3
        self.Adj_Graph=Adj_Graph
        self.epsilon_g=epsilon_g
        self.gamma_c=gamma_c
        self.gamma_g=gamma_g
        self.GMM_weight = GMM_weight
        print("AMM_no_dag init model.")

        if emb_mat is None:
            self.Net = net(
                max_topic_num,
                batch_size,
                topic_num=self.topic_num_3,
                Adj_Graph=self.Adj_Graph,
                epsilon_g = self.epsilon_g,
                gamma_c = self.gamma_c,
                gamma_g = self.gamma_g,
                GMM_weight=self.GMM_weight,
                **kwargs,
            ).to(device)
        else:
            emb_mat = torch.from_numpy(emb_mat.astype(np.float32)).to(device)
            self.Net = net(
                max_topic_num,
                batch_size,
                topic_num=self.topic_num_3,
                emb_mat=emb_mat.T,
                Adj_Graph=self.Adj_Graph,
                epsilon_g=epsilon_g,
                gamma_c=gamma_c,
                gamma_g=gamma_g,
                GMM_weight=GMM_weight,
                **kwargs,
            ).to(device)

        print(self.Net)

        self.max_topic_num = max_topic_num
        self.pi_ave = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate



        # optimizer uses ADAM

    def initalize_A(self, topic_nums=16):
        A = np.ones([topic_nums, topic_nums]) / (topic_nums - 1) + (
            np.random.rand(topic_nums * topic_nums) * 0.0002
        ).reshape([topic_nums, topic_nums])
        for i in range(topic_nums):
            A[i, i] = 0
        return A

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.Net.state_dict(), f"{self.model_path}/model.pkl")
        with open(f"{self.model_path}/topic_num.txt", "w") as f:
            f.write(str(self.max_topic_num))
        np.save(f"{self.model_path}/pi_ave.npy", self.pi_ave)
        print(f"Models save to  {self.model_path}/model.pkl")

    def load_model(self, model_filename="model.pkl"):
        model_path = os.path.join(self.model_path, model_filename)

        self.Net.load_state_dict(torch.load(model_path))
        # self.Net = torch.load(model_path)
        with open(f"{self.model_path}/topic_num.txt", "r") as f:
            self.max_topic_num = int(f.read())
        self.pi_ave = np.load(f"{self.model_path}/pi_ave.npy")
        print("AMM_no_dag model loaded from {}.".format(model_path))


    def get_word_topic(self, data):
        word_topic = self.Net.infer(torch.from_numpy(data).to(device))
        word_topic = self.to_np(word_topic)
        return word_topic

    def get_topic_dist(self,):
        # topic_dist = self.Net.get_topic_dist()[self.topics]
        topic_dist = self.Net.get_topic_dist()
        return topic_dist

    def get_topic_word(self, top_k=15, vocab=None):
        topic_dist = self.get_topic_dist()
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [
            [self.reader.vocab[idx] for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]
        return topic_words

    def get_topic_parents(self, mat):
        return 0

    def evaluate(self):
        # 重定向回文件
        _, _, texts,context = self.reader.get_sequence("all")

        topic_word = self.get_topic_word(
            top_k=10, vocab=self.reader.vocab
        )
        # 打印top N的主题词
        for k, top_word_k in enumerate(topic_word):
            print(f"Topic {k}:{top_word_k}")


    # NPMI
    def sampling(self, flag, data, exp_ini):
        print("experiment setting: ", exp_ini)
        self.Net.eval()
        # 计算coherence，祖传方法
        (test_data, test_label, _)  = self.reader.get_matrix(data, mode="count")
        topic_dist = self.to_np(self.get_topic_dist())  # 最低层主题的 coherence
        train_coherence = utils.evaluate_NPMI(test_data, topic_dist)
        CV = utils.evaluate_CV(test_data, topic_dist)
        pre_embedding=self.pre_embeddings
       # print('pre_embedding',pre_embedding)
        # WE = utils.evaluate_WE(test_data, topic_dist,pre_embedding,n_list=[5,10,15])
        TU = utils.evaluate_TU(topic_dist)
            # TU2 = utils.evaluate_TU(topic_dist_2)
            # TU1 = utils.evaluate_TU(topic_dist_1)
            # TU0 = utils.evaluate_TU(topic_dist_0)
            # TU = utils.evaluate_TU(topic_dist            
           # print("Topic coherence:", train_coherence)
            #score = 2 * (train_coherence - best_coh) / best_coh + (TU - best_TU) / best_TU
    #    print("Topic coherence  res: ", train_coherence_res)
        TQ = TU * train_coherence
        print("Topic coherence : ", train_coherence)
        print("TU: " + str(TU))
        print("TQ:", TQ)
        print("Topic coherence:", train_coherence)
        print('CV:',CV)
        # print('WE:',WE)
      #  if  score> self.best_coherence:
        if TQ > self.best_score:
            self.best_score = TQ
            print("New best_score found!!  is", self.best_score)
            self.save_model()

        if exp_ini is not None:

            (test_data, test_label, test_text, test_context)  = self.reader.get_sparse_matrix('test', mode = "tfidf")
            (train_data, train_label, train_text, train_context)  = self.reader.get_sparse_matrix('train+valid', mode = "tfidf")
            test_data = torch.from_numpy(test_data.toarray()).to(device)
            test_context = torch.from_numpy(np.array(test_context)).to(device)
        
            train_data = torch.from_numpy(train_data.toarray()).to(device)
            train_context = torch.from_numpy(np.array(train_context)).to(device)
            train_tf_theta, train_context_theta, train_theta = self.Net.get_theta(train_data, train_context)
            test_tf_theta, test_context_theta, test_theta = self.Net.get_theta(test_data, test_context)

            save_dir = os.path.join("./cg_res", exp_ini)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        #    print('save_dir,',save_dir)
            np.save(os.path.join(save_dir, "topic_word.npy"), topic_dist)
            np.save(os.path.join(save_dir, "train_theta.npy"), self.detach_np(train_theta))
            np.save(os.path.join(save_dir, "train_tf_theta.npy"), self.detach_np(train_tf_theta))
            np.save(os.path.join(save_dir, "train_context_theta.npy"), self.detach_np(train_context_theta))
            np.save(os.path.join(save_dir, "test_theta.npy"), self.detach_np(test_theta))
            np.save(os.path.join(save_dir, "test_tf_theta.npy"), self.detach_np(test_tf_theta))
            np.save(os.path.join(save_dir, "test_context_theta.npy"), self.detach_np(test_context_theta))

            model_list = ['CGTM']  # ['CGTM', 'ZeroShotTM', 'CombinedTM', 'ECRTM', 'WeTe', 'GNTM', 'NSTM', 'GINopic', 'ETM', 'ProdLDA', 'LDA']
            dataset = exp_ini.split('_')[0]
            for model in model_list:
                    print("Evaluating model: {} on dataset {}".format(model, dataset))
                    n_list = [5, 10, 15]
                    doc_word, vocab, train_labels, test_labels, train_doc_topic, test_doc_topic, topic_word = evaluate.load_res(
                        model, dataset, exp_ini)
                    n_labels = np.max(train_labels) + 1
                    print("n_labels: {}".format(n_labels))
                    res_topic = evaluate.topic_evaluation(doc_word, topic_word, vocab, n_list)
                    print("Topic Result: ", res_topic, "\n")
                    evaluate.save_res(model, exp_ini, res_topic, "topic_res.json")
                    res_doc = evaluate.doc_evaluation(train_doc_topic, test_doc_topic, train_labels, test_labels, n_labels)
                    print("Document Result: ", res_doc, "\n")
                    evaluate.save_res(model, exp_ini, res_doc, "doc_res.json")
            

    def get_batches(self, batch_size=300, rand=True):
        n, d = self.train_data.shape

        batchs = n // batch_size
        while True:
            idxs = np.arange(self.train_data.shape[0])

            if rand:
                np.random.shuffle(idxs)

            for count in range(batchs):
                wordcount = []
                beg = count * batch_size
                end = (count + 1) * batch_size

                idx = idxs[beg:end]
                data = self.train_data[idx].toarray()
                data = torch.from_numpy(data).to(device)

                context=self.context[idx]
                context = torch.from_numpy(context).to(device)
             #   print(data,context)
                yield [data,context]

    def get_topic_word_new(self, top_k=15, vocab=None):
        topic_dist = self.get_topic_dist()
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [
            [self.reader.vocab[idx] for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]
        words_embedding = [
            [self.Net.GIA_emb.T[idx].detach().to("cpu").numpy() for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]

        return topic_words, words_embedding,


    def train(self, epochs=320, batch_size=256, data_type="train+valid", exp_ini = None):
        self.t_begin = time.time()
        batch_size = self.batch_size
        (
            self.train_data,
            self.train_label,
            self.train_text,
            self.context,
        ) = self.reader.get_sparse_matrix(data_type, mode="tfidf")#count
        print('self.train_text', len(self.train_text))#分词好的
        print(self.context.shape)
        print( self.train_data.shape)
        self.train_generator = self.get_batches(batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // batch_size
        print(batch_size)
        self.best_coherence = -1
        self.best_score= -1
        optimizer = optim.RMSprop(self.Net.parameters(), lr=self.lr)
        optimizer2 = optim.RMSprop(
            [self.Net.topic_embed], lr=self.lr * 0.2
        )
        clipper = WeightClipper(frequency=1)
        for epoch in tqdm(range(self.epochs)):

            self.Net.train()
            epoch_word_all = 0
            doc_count = 0

            # if epoch % (3) < 1:  #
            #     self.Net.topic_embed.requires_grad = False
            #
            # else:
            #     self.Net.topic_embed.requires_grad = True

            for i in range(n_batchs):
                optimizer.zero_grad()
                optimizer2.zero_grad()
                temperature = max(0.95 ** epoch, 0.5)
                [ori_docs,context] = next(self.train_generator)
               # ori_text = next(self.train_generator)
                doc_count += ori_docs.shape[0]
                count_batch = []
                for idx in range(ori_docs.shape[0]):
                    count_batch.append(np.sum(self.to_np(ori_docs[idx])))

                epoch_word_all += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)

                loss = self.Net(
                    ori_docs,context, temperature = temperature
                )
                loss.backward()


                optimizer.step()



            self.Net.eval()
            # if epoch == self.epochs - 1:
            #      self.save_model()
            if epoch > 0 and (epoch + 1) % 10 == 0:
                self.sampling(flag = 1, data='test', exp_ini = exp_ini)


        self.t_end = time.time()
        print("Time of training-{}".format((self.t_end - self.t_begin)))

    def detach_np(self, x):
        return x.cpu().detach().numpy()  


    def test(self, exp_ini):
        self.load_model()
        self.Net.eval()
        self.best_coherence = 999
        self.best_score = 999
        self.evaluate()
        self.sampling(flag = 1, data = 'test', exp_ini = exp_ini)
        #self.eval_level()

    def get_params(self, batch_size = 256, data_type = 'train+valid'):
           #     print('111',dataset)
        batch_size = self.batch_size
        (
            self.train_data,
            self.train_label,
            self.train_text,
            self.context,
        ) = self.reader.get_sparse_matrix(data_type, mode="tfidf")#count

        print('self.train_text', len(self.train_text))#分词好的
        print(self.context.shape)
        print( self.train_data.shape)
        self.train_generator = self.get_batches(batch_size)
        data_size = self.train_data.shape[0]
        n_batchs = data_size // batch_size
        print(batch_size)
        self.best_coherence = -1
        self.best_score= -1
        optimizer = optim.RMSprop(self.Net.parameters(), lr=self.lr)
        optimizer2 = optim.RMSprop(
            [self.Net.topic_embed], lr=self.lr * 0.2
        )
        clipper = WeightClipper(frequency=1)
        self.t_begin = time.time()
        for epoch in tqdm(range(1)):

            self.Net.train()
            epoch_word_all = 0
            doc_count = 0

            for i in range(n_batchs):
                optimizer.zero_grad()
                optimizer2.zero_grad()
                temperature = max(0.95 ** epoch, 0.5)
                [ori_docs,context] = next(self.train_generator)
               # ori_text = next(self.train_generator)
                doc_count += ori_docs.shape[0]
                count_batch = []
                for idx in range(ori_docs.shape[0]):
                    count_batch.append(np.sum(self.to_np(ori_docs[idx])))

                epoch_word_all += np.sum(count_batch)
                count_batch = np.add(count_batch, 1e-12)

                loss1 = self.Net(
                    ori_docs,context, temperature = temperature
                )
                if i == 0:
                    flops, params = profile(self.Net, inputs = (ori_docs, context, None, temperature))    
                    print('flops:{}'.format(flops))
                    print('params:{}'.format(params))
                loss1.backward()

                optimizer.step()
            self.Net.eval()


        self.t_end = time.time()
        print("Time of training-{}".format((self.t_end - self.t_begin)))

def main(exp_ini = '20news',
         mode = 'Train',
        dataset = "20news",
        max_topic_num = 300,
        emb_type = "glove",
        epsilon_g = 2,
        gamma_c = 2,
        gamma_g = 2,
        GMM_weight = 2,
        adj_alpha = 2,
        **kwargs):

 #   base_path = os.path.expanduser('~') + '/Methods/HNTM and nHNTM/'
    data_path = f"./data/{dataset}"
    reader = TextReader(data_path)
    print(emb_type)
    if emb_type == "bert":
        bert_emb_path = f"./emb/bert.npy"
        embedding_mat = utils.build_bert_embedding(bert_emb_path, reader.vocab,
                                                   data_path)
    elif emb_type == "glove":
        emb_path = f"./emb/glove.6B.300d.txt"
        embedding_mat = utils.build_embedding(emb_path, reader.vocab,
                                              data_path)[:-1]
    else:
        embedding_mat = None
   # print('embedding_mat',embedding_mat,embedding_mat.shape)
    model_path = f'./model/{exp_ini}_{max_topic_num}_{reader.vocab_size}'
    print('adj_alpha', adj_alpha)
    print( "epsilon_g: ", epsilon_g,
        " gamma_c:" , gamma_c,
        "gamma_g", gamma_g,
        "GMM_weight", GMM_weight)
    A_C = Co_Graph(reader, data_type="train+valid",adj_alpha=adj_alpha,device=device)
    A_C = convert_to_tensor(A_C, device)    #WR    一致

    model = AMM_no_dag(reader, max_topic_num, model_path, embedding_mat, epsilon_g=epsilon_g,gamma_c=gamma_c,gamma_g=gamma_g,GMM_weight=GMM_weight,Adj_Graph=[A_C],**kwargs)
   # print('11',dataset)
    if mode == 'Train':
        model.train(epochs = 320, batch_size = 256, data_type = "train+valid", exp_ini = exp_ini)
    elif mode == 'Test':
        model.test(exp_ini = exp_ini)
    elif mode == 'Continual':
        model.load_model()
        model.train(epochs = 320, batch_size = 256, data_type = "train+valid", exp_ini = exp_ini)
        model.test(exp_ini = exp_ini)
    elif mode == 'GP':
        model.get_params(batch_size = 256, data_type = "train+valid")
    else:
        print(f'Unknowned mode {mode}!') 


if __name__ == '__main__':
    exp_ini = sys.argv[1]
    # print(exp_ini)
    #dataset
    config = yaml.load(open('config.yaml'), yaml.FullLoader)
    # if config['para']['dataset']=="nips":
    #     best_coh=0.142
    #     best_TU=0.68
    # elif config['para']['dataset']=="20news":
    #     best_coh = 0.294
    #     best_TU = 0.820
    #main(mode="dp", **config[dataset])
    main(exp_ini = exp_ini, mode = "Train", **config[exp_ini])
    # main(exp_ini = exp_ini, mode = "Test", **config[exp_ini])
    # main(exp_ini = exp_ini, mode = "GP", **config[exp_ini])
    # parameter_idx = int(sys.argv[1])
    # parameters = ['epsilon_g', 'gamma_c', 'gamma_g', 'GMM_weight']

    # parameter_grid = {
    # 'epsilon_g': [0.001, 0.004, 0.01, 0.1, 0.4],
    # 'gamma_c': [0.1, 0.5, 1, 2, 4],
    # 'gamma_g': [0.1, 0.2, 0.5, 1, 2],
    # 'GMM_weight': [5, 10, 20, 40, 80]
    # }
    # parameter = parameters[parameter_idx]

    # for value in parameter_grid[parameter]:
    #     exp_ini = '20news_{}_{}'.format(parameter, value)
    #     main(exp_ini = exp_ini, mode = "Train", **config[exp_ini])
    #     main(exp_ini = exp_ini, mode = "Test", **config[exp_ini])
    # main(mode="Continual", **config[dataset])


