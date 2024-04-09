from pathlib import Path
import os
import sys
import glob
from joblib import dump, load
import pandas as pd
import scipy
import scipy.stats
import scipy.special
import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils.riesznet import RieszNetPrime, RieszNetNonPrime
# from utils.moments import ate_moment_fn
from utils.moments import recant_nonprime_moment_fn_1, recant_nonprime_moment_fn_2, recant_nonprime_moment_fn_3
from utils.moments import recant_prime_moment_fn_1_Z, recant_prime_moment_fn_1_M, recant_prime_moment_fn_2_Z, recant_prime_moment_fn_2_M, recant_prime_moment_fn_3_Z, recant_prime_moment_fn_4

drop_prob = 0.0  # dropout prob of dropout layers throughout notebook
n_hidden = 100  # width of hidden layers throughout notebook

# Training params
learner_lr = 1e-5
learner_l2 = 0
learner_l1 = 0.0
n_epochs = 600
earlystop_rounds = 40 # how many epochs to wait for an out-of-sample improvement
earlystop_delta = 1e-4
target_reg_1 = 0
target_reg_2 = 0
target_reg_3 = 0
target_reg_4 = 0
target_reg_5 = 0
target_reg_6 = 0
riesz_weight_1 = 1
riesz_weight_2 = 1
riesz_weight_3 = 0
riesz_weight_4 = 0
riesz_weight_5 = 0
riesz_weight_6 = 0


bs = 64
device = torch.cuda.current_device() if torch.cuda.is_available() else None
print("GPU:", torch.cuda.is_available())

from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r

def _combinations(n_features, degree, interaction_only):
        comb = (combinations if interaction_only else combinations_w_r)
        return chain.from_iterable(comb(range(n_features), i)
                                   for i in range(0, degree + 1))

class Learner(nn.Module):

    def __init__(self, n_t, n_hidden, p, degree, interaction_only=False):
        super().__init__()
        n_common = 200
        
        self.common1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 4, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())
        self.common2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 4, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())
        self.common3 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 2, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())
        self.common4 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 2, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())
        self.common5 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 2, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())
        self.common6 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ReLU())


        self.riesz_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn3 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn4 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn5 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_nn6 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))

    def forward(self, x):
        # Create a new dataset x1, which is (A, W)
        # Create a new dataset x2, which is (A, Z, W)
        # Create a new dataset x3, which is (A, M, W)
        # Recall: we assume x = (A, Z, M, W)

        if torch.is_tensor(x):
            with torch.no_grad():
                x1 = torch.cat([torch.reshape(x[:, 0].to(device), (-1, 1)), x[:, 5:]], dim=1)
        else:
            x1 = np.hstack([np.array(x[:, 0]).reshape(-1, 1), x[:, 5:]])
        
        if torch.is_tensor(x):
            with torch.no_grad():
                x2 = torch.cat([torch.reshape(x[:, [0, 1, 2]].to(device), (-1, 3)), x[:, 5:]], dim=1)
        else:
            x2 = np.hstack([np.array(x[:, [0, 1, 2]]).reshape(-1, 3), x[:, 5:]])
        
        if torch.is_tensor(x):
            with torch.no_grad():
                x3 = torch.cat([torch.reshape(x[:, [0, 3, 4]].to(device), (-1, 3)), x[:, 5:]], dim=1)
        else:
            x3 = np.hstack([np.array(x[:, [0, 3, 4]]).reshape(-1, 3), x[:, 5:]])

        feats1 = self.common1(x1)
        feats2 = self.common2(x1)
        feats3 = self.common3(x2)
        feats4 = self.common4(x2)
        feats5 = self.common5(x3)
        feats6 = self.common6(x)

        riesz1 = self.riesz_nn1(feats1) #+ self.riesz_poly1(poly1)
        riesz2 = self.riesz_nn2(feats2) #+ self.riesz_poly2(poly1)
        riesz3 = self.riesz_nn3(feats3) #+ self.riesz_poly3(poly2)
        riesz4 = self.riesz_nn4(feats4) #+ self.riesz_poly4(poly3)
        riesz5 = self.riesz_nn5(feats5) #+ self.riesz_poly5(poly4)
        riesz6 = self.riesz_nn6(feats6) #+ self.riesz_poly5(poly4)

        return torch.cat([riesz1, riesz2, riesz3, riesz4, riesz5, riesz6], dim=1)

moment_fn_1 = recant_prime_moment_fn_1_Z
moment_fn_2 = recant_prime_moment_fn_1_M
moment_fn_3 = recant_prime_moment_fn_2_Z
moment_fn_4 = recant_prime_moment_fn_2_M
moment_fn_5 = recant_prime_moment_fn_3_Z
moment_fn_6 = recant_prime_moment_fn_4

N = 2000
label = os.environ.get('SLURM_ARRAY_TASK_ID')

filename = "~/Projects/RieszLearning/make_data_nonnull_" + str(N) + "/simulation_test_all_batches_" + str(N) + "_cont_multi_Z_M_Y_recantingtwins_array_" + str(label) + ".csv"
data_batch_1 = pd.read_csv(filename, sep=',', encoding='utf-8')

filename = "~/Projects/RieszLearning/make_data_nonnull_" + str(N) + "/simulation_test_all_batches_" + str(N) + "_cont_multi_Z_M_Y_recantingtwins_supp_Z_array_" + str(label) + ".csv"

data_batch_1_supp_Z = pd.read_csv(filename, sep=',', encoding='utf-8')

A = np.array(data_batch_1["A"])
Z_1 = np.array(data_batch_1["Z_1"])
Z_2 = np.array(data_batch_1["Z_2"])
M_1 = np.array(data_batch_1["M_1"])
M_2 = np.array(data_batch_1["M_2"])
W_1 = np.array(data_batch_1["W_1"])
W_2 = np.array(data_batch_1["W_2"])
W_3 = np.array(data_batch_1["W_3"])
Y = np.array(data_batch_1["Y"])

A_supp = np.array(data_batch_1_supp_Z["A"])
Z_1_supp = np.array(data_batch_1_supp_Z["Z_1"])
Z_2_supp = np.array(data_batch_1_supp_Z["Z_2"])
M_1_supp = np.array(data_batch_1_supp_Z["M_1"])
M_2_supp = np.array(data_batch_1_supp_Z["M_2"])
W_1_supp = np.array(data_batch_1_supp_Z["W_1"])
W_2_supp = np.array(data_batch_1_supp_Z["W_2"])
W_3_supp = np.array(data_batch_1_supp_Z["W_3"])
Y_supp = np.array(data_batch_1_supp_Z["Y"])

X = np.c_[A, Z_1, Z_2, M_1, M_2, W_1, W_2, W_3]
X_supp = np.c_[A_supp, Z_1_supp, Z_2_supp, M_1_supp, M_2_supp, W_1_supp, W_2_supp, W_3_supp]

X_train = X.copy()
y_train = Y.copy()
X_test = X.copy()
y_test = Y.copy()
X_train_supp = X_supp.copy()
X_test_supp = X_supp.copy()

torch.cuda.empty_cache()
learner = Learner(X_train.shape[1], n_hidden, drop_prob, 0, interaction_only=True)
agmm = RieszNetPrime(learner, moment_fn_1, moment_fn_2, moment_fn_3, moment_fn_4, moment_fn_5, moment_fn_6, a_1 = 0, a_2 = 0, a_3 = 1, a_4 = 1)

# Fast training

riesz_weight_1 = 1
riesz_weight_2 = 1
riesz_weight_3 = 0
riesz_weight_4 = 0
riesz_weight_5 = 0
riesz_weight_6 = 0

agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-03, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=100, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam',
          model_dir=str(Path.home()), device=device, verbose=1)

# riesz_weight_1 = 0
# riesz_weight_2 = 0.1

# Fine tune
agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=50, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

# Freeze network parameters w.r.t. alpha1

for name, param in agmm.learner.named_parameters():
    if("riesz_nn1" in name or "common1" in name):
        print(name)
        param.requires_grad = False
    
# Freeze network parameters w.r.t. alpha1

for name, param in agmm.learner.named_parameters():
    if("riesz_nn2" in name or "common2" in name):
        print(name)
        param.requires_grad = False

riesz_weight_1 = 0
riesz_weight_2 = 0
riesz_weight_3 = 1
riesz_weight_4 = 1
riesz_weight_5 = 0
riesz_weight_6 = 0

agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-03, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=100, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

# Fine tune
agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=50, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

# riesz_weight_1 = 0
# riesz_weight_2 = 0.1

# Fine tune
# agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
#           earlystop_rounds=2, earlystop_delta=earlystop_delta,
#           learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
#           n_epochs=100, bs=bs, target_reg_1=target_reg_1,
#           riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
#           riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
#           riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
#           riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
#           riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
#           riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
#           model_dir=str(Path.home()), device=device, verbose=1)

# Freeze network parameters w.r.t. alpha1

for name, param in agmm.learner.named_parameters():
    if("riesz_nn3" in name or "common3" in name):
        print(name)
        param.requires_grad = False
    
# Freeze network parameters w.r.t. alpha1

for name, param in agmm.learner.named_parameters():
    if("riesz_nn4" in name or "common4" in name):
        print(name)
        param.requires_grad = False
        
riesz_weight_1 = 0
riesz_weight_2 = 0
riesz_weight_3 = 0
riesz_weight_4 = 0
riesz_weight_5 = 1
riesz_weight_6 = 0

agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-03, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=100, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

# Fine tune
agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=50, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

for name, param in agmm.learner.named_parameters():
    if("riesz_nn5" in name or "common5" in name):
        print(name)
        param.requires_grad = False

riesz_weight_1 = 0
riesz_weight_2 = 0
riesz_weight_3 = 0
riesz_weight_4 = 0
riesz_weight_5 = 0
riesz_weight_6 = 1

agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-03, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=100, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

# Fine tune
agmm.fit(X_train, y_train, X_train_supp, Xval=X_test, yval=y_test, Xval_supp_Z = X_test_supp,
          earlystop_rounds=2, earlystop_delta=earlystop_delta,
          learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
          n_epochs=50, bs=bs, target_reg_1=target_reg_1,
          riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
          riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3,
          riesz_weight_3=riesz_weight_3, target_reg_4=target_reg_4,
          riesz_weight_4=riesz_weight_4, target_reg_5=target_reg_5,
          riesz_weight_5=riesz_weight_5, target_reg_6=target_reg_6,
          riesz_weight_6=riesz_weight_6, optimizer='adam', warm_start=True,
          model_dir=str(Path.home()), device=device, verbose=1)

alpha_results = pd.DataFrame(data={"A": A, "Z_1": Z_1, "Z_2": Z_2, "M_1": M_1, "M_2": M_2, "W_1": W_1, "W_2": W_2, "W_3": W_3, "Y": Y,  "alpha1": agmm.predict(X)[:,0], "alpha2": agmm.predict(X)[:,1], "alpha3": agmm.predict(X)[:,2], "alpha4": agmm.predict(X)[:,3], "alpha5": agmm.predict(X)[:,4], "alpha6": agmm.predict(X)[:,5], "label": int(label)})

alpha_results.to_csv("/gpfs/home/ll4245/Projects/RieszLearning/results_0011/N_%d_seq_%d.csv" % (N, int(label)))
