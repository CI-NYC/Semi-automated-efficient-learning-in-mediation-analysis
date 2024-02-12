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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.riesznet import RieszNet, RieszNetNDE
# from utils.moments import ate_moment_fn
from utils.moments import nde_theta1_moment_fn, nde_theta2_moment_fn
# from utils.ihdp_data import *

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

        # Don't know what it is, but should be different for 1 and 2
        self.monomials1 = list(_combinations(n_t - 1, degree, interaction_only))
        self.monomials2 = list(_combinations(n_t, degree, interaction_only))

        # Common layers for g and alpha
        # theta_1 and m_1 are the function of (A, W)
        # theta_2 and m_2 are the function of (A, M, W)
        self.common1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t - 2, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())
        self.common2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_common, n_common), nn.ELU())

        # Riesz specific layers
        # alpha_1 is related to theta_1 and m_1
        # alpha_2 is related to theta_2 and m_2
        self.riesz_nn1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly1 = nn.Sequential(nn.Linear(len(self.monomials1), 1))
        self.riesz_nn2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, 1))
        self.riesz_poly2 = nn.Sequential(nn.Linear(len(self.monomials2), 1))

        # Regression loss layers
        # Indexes are the same as Riesz specific layers
        self.reg_nn0_1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn1_1 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn0_2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        self.reg_nn1_2 = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_common, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, n_hidden), nn.ELU(),
                                    nn.Dropout(p=p), nn.Linear(n_hidden, 1))

        # Don't know what it is, but should be different for 1 and 2
        self.reg_poly1 = nn.Sequential(nn.Linear(len(self.monomials1), 1))
        self.reg_poly2 = nn.Sequential(nn.Linear(len(self.monomials2), 1))


    def forward(self, x):
        # Create a new dataset x1, which is (A, W), used for theta_1 and m_1
        # Recall: we assume x = (A, M, W)

        if torch.is_tensor(x):
            with torch.no_grad():
                x1 = torch.cat([torch.reshape(x[:, 0].to(device), (-1, 1)), x[:, 3:]], dim=1)
        else:
            x1 = np.hstack([np.array(x[:, 0]).reshape(-1, 1), x[:, 3:]])

        poly1 = torch.cat([torch.prod(x1[:, t], dim=1, keepdim=True)
                          for t in self.monomials1], dim=1)
        poly2 = torch.cat([torch.prod(x[:, t], dim=1, keepdim=True)
                          for t in self.monomials2], dim=1)

        feats1 = self.common1(x1)
        feats2 = self.common2(x)

        riesz1 = self.riesz_nn1(feats1) + self.riesz_poly1(poly1)
        riesz2 = self.riesz_nn2(feats2) + self.riesz_poly2(poly2)

        reg1 = self.reg_nn0_1(feats1) * (1 - x1[:, [0]]) + self.reg_nn1_1(feats1) * x1[:, [0]] + self.reg_poly1(poly1)
        reg2 = self.reg_nn0_2(feats2) * (1 - x[:, [0]]) + self.reg_nn1_2(feats2) * x[:, [0]] + self.reg_poly2(poly2)
        return torch.cat([reg1, riesz1, reg2, riesz2], dim=1)

if __name__ == '__main__':
    # data = pd.read_csv("/gpfs/home/ll4245/Projects/Ivan_Diaz/RieszLearning/data/RR_NDE_Simulation_Data.csv")
    # data = data.iloc[:,1:]
    label = sys.argv[1]
    
    N = 5000

    # Will be (W_1, W_2)
    W = scipy.stats.multivariate_normal.rvs(mean = np.array([0, 0]), cov = 1, size = N)

    W_1 = W[:, 0]
    W_2 = W[:, 1]

    prob_A = scipy.special.expit(0.5 * np.sin(W[:, 0]) + 0.7 * np.cos(W[:, 1]))
    A = scipy.stats.bernoulli.rvs(prob_A, size=N)

    # Will be (M_1, M_2)
    M_1 = A + 0.2 * W[:, 0] ** 2
    M_2 = np.abs(W[:, 1]) + A

    M_1 = scipy.stats.norm.rvs(loc = M_1, scale = 1, size = N)
    M_2 = scipy.stats.norm.rvs(loc = M_2, scale = 1, size = N)
    M = np.concatenate([M_1.reshape(-1, 1), M_2.reshape(-1, 1)], axis= 1)

    M_1 = M[:, 0]
    M_2 = M[:, 1]

    Y = scipy.stats.norm.rvs(loc = np.abs(W[:, 0]) + A + M_1 + np.cos(M_2), scale = 1, size = N)
    
    data_batch_1 = pd.DataFrame(data = {"W_1": W_1, "W_2": W_2, "A": A, "M_1": M_1, "M_2": M_2, "Y": Y})

    moment_fn_1 = nde_theta1_moment_fn
    moment_fn_2 = nde_theta2_moment_fn
    # moment_fn = ate_moment_fn
    
    drop_prob = 0.0  # dropout prob of dropout layers throughout notebook
    n_hidden = 100  # width of hidden layers throughout notebook

    # Training params
    learner_lr = 1e-5
    learner_l2 = 1e-3
    learner_l1 = 0.0
    n_epochs = 600
    earlystop_rounds = 40 # how many epochs to wait for an out-of-sample improvement
    earlystop_delta = 1e-4
    target_reg_1 = 0
    target_reg_2 = 0
    riesz_weight_1 = 1
    riesz_weight_2 = 0

    bs = 64
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    X = np.c_[A, M_1, M_2, W_1, W_2]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

    torch.cuda.empty_cache()
    learner = Learner(X_train.shape[1], n_hidden, drop_prob, 0, interaction_only=True)
    agmm = RieszNetNDE(learner, moment_fn_1, moment_fn_2)
    
    # Fast training
    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
              earlystop_rounds=2, earlystop_delta=earlystop_delta,
              learner_lr=1e-04, learner_l2=learner_l2, learner_l1=learner_l1,
              n_epochs=100, bs=bs, target_reg_1=target_reg_1,
              riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
              riesz_weight_2=riesz_weight_2, optimizer='adam',
              model_dir=str(Path.home()), device=device, verbose=1)
    
    # riesz_weight_1 = 0
    # riesz_weight_2 = 0.1

    # Fine tune
    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
              earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
              learner_lr=learner_lr, learner_l2=learner_l2, learner_l1=learner_l1,
              n_epochs=100, bs=bs, target_reg_1=target_reg_1,
              riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
              riesz_weight_2=riesz_weight_2, optimizer='adam', warm_start=True,
              model_dir=str(Path.home()), device=device, verbose=1)
    
    # Freeze network parameters w.r.t. alpha1

    for name, param in agmm.learner.named_parameters():
        if("riesz_nn1" in name or "riesz_poly1" in name or "common1" in name):
            param.requires_grad = False
            
    # Fast training
    riesz_weight_2 = 1
    riesz_weight_1 = 0

    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
              earlystop_rounds=2, earlystop_delta=earlystop_delta,
              learner_lr=1e-4, learner_l2=learner_l2, learner_l1=learner_l1,
              n_epochs=100, bs=bs, target_reg_1=target_reg_1,
              riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
              riesz_weight_2=riesz_weight_2, optimizer='adam',
              model_dir=str(Path.home()), device=device, verbose=1, warm_start=True)
    
    # riesz_weight_1 = 0
    # riesz_weight_2 = 0.1

    # Fine tune
    agmm.fit(X_train, y_train, Xval=X_test, yval=y_test,
              earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
              learner_lr=1e-05, learner_l2=learner_l2, learner_l1=learner_l1,
              n_epochs=100, bs=bs, target_reg_1=target_reg_1,
              riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
              riesz_weight_2=riesz_weight_2, optimizer='adam', warm_start=True,
              model_dir=str(Path.home()), device=device, verbose=1)
    
    methods = ['dr', 'direct', 'ips', 'test']
    srr = {'dr' : True, 'direct' : False, 'ips' : True, 'test': True}
    
    res = agmm.predict_avg_moment(X, Y,  model='earlystop', method = "test", srr = srr["test"])[0]
    res_u = agmm.predict_avg_moment(X, Y,  model='earlystop', method = "test", srr = srr["test"])[2]
    res_l = agmm.predict_avg_moment(X, Y,  model='earlystop', method = "test", srr = srr["test"])[1]
    
    # True Value for Estimator X = np.c_[A, M_1, M_2, W_1, W_2]
    
    prob_A = 1 / (1 + np.exp(-0.5 * np.sin(X[:, 3]) - 0.7 * np.cos(X[:, 4])))
    alpha1 = (1 / (1 - prob_A)) * (X[:,0] == 0)

    alpha2_part1 = (X[:,0] == 1) * (1 / (prob_A))
    alpha2_ratio_1 = scipy.stats.norm.pdf(X[:,1], loc = 0.2 * (X[:,3] ** 2), scale = 1) / scipy.stats.norm.pdf(X[:,1], loc = 0.2 * (X[:,3] ** 2) + 1, scale = 1)
    alpha2_ratio_2 = scipy.stats.norm.pdf(X[:,2], loc = np.abs(X[:, 4]), scale = 1) / scipy.stats.norm.pdf(X[:,2], loc = np.abs(X[:, 4]) + 1, scale = 1)
    alpha2 = alpha2_part1 * alpha2_ratio_1 * alpha2_ratio_2 - alpha1

    X_transformed = X.copy()
    X_transformed[:, 2] = np.cos(X_transformed[:, 2])
    X_transformed[:, 3] = np.abs(X_transformed[:, 3])

    model = LinearRegression().fit(X_transformed, Y)
    Y_pred = model.predict(X_transformed)

    X_sub_0 = X_transformed.copy()
    X_sub_0[:,0] = 0

    X_sub_1 = X_transformed.copy()
    X_sub_1[:,0] = 1

    pseudo_Y = model.predict(X_sub_1) - model.predict(X_sub_0)

    X_remove_M = X.copy()
    X_remove_M = X_remove_M[:, [0, 3, 4]]

    X_remove_M[:, 1] = X_remove_M[:, 1] ** 2
    X_remove_M[:, 2] = np.abs(X_remove_M[:, 2])

    model_2 = LinearRegression().fit(X_remove_M, pseudo_Y)

    X_remove_M_sub_0 = X_remove_M.copy()
    X_remove_M_sub_0[:,0] = 0

    pseudo_Y_pred = model_2.predict(X_remove_M_sub_0)
    
    estimator = np.mean(alpha2 * (Y - Y_pred) + alpha1 * (pseudo_Y - pseudo_Y_pred) + pseudo_Y_pred)
    
    estimator_2 = np.mean(pseudo_Y_pred)
    
    summary = pd.DataFrame(data = {"true": estimator, "true_pseudo_Y": estimator_2, "pred": res, "pred_upper": res_u, "pred_lower": res_l}, index=[0])
    
    summary.to_csv("/gpfs/home/ll4245/Projects/Ivan_Diaz/RieszLearning/results/DGP_3_N_%d_seq_%d.csv" % (N, int(label)))
    
    
    

