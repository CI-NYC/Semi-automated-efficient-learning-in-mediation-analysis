import os
import copy
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
import scipy.stats
import statsmodels.api as sm

from scipy.spatial import distance_matrix
from cvxopt import spmatrix, matrix
from cvxopt import solvers

solvers.options['show_progress'] = False
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

def mean_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def L1_reg(net, l1_value, skip_list=()):
    L1_reg_loss = 0.0
    for name, param in net.named_parameters():
        if not param.requires_grad or len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            continue  # frozen weights
        else:
            L1_reg_loss += torch.sum(abs(param))
    L1_reg_loss *= l1_value
    return L1_reg_loss

def permutation_matrix(data_temp, device):
    # A, Z, M, W1, W2, W3
    data_temp = data_temp.cpu().detach().numpy()
    N = np.shape(data_temp)[0]
    D = distance_matrix(data_temp[:, [0, 3]].reshape(-1, 2), data_temp[:, [0, 3]].reshape(-1, 2))  # .reshape(-1, )
    row, col = np.triu_indices(N, 1)
    D[row, col] = 0

    D = D.reshape(-1, )
    D = D / max(D)
    D = matrix(D)

    A2 = -np.eye(N * N)
    b2 = np.zeros(N * N)

    row_list = []
    col_list = []
    data_list = []
    for i in range(N):
        index_col_list = [i * N + j for j in range(N)]
        col_list = col_list + index_col_list
        row_list = row_list + [i for j in range(N)]
        data_list = data_list + [1 for j in range(N)]

    for i in range(N - 1):
        index_col_list = [j * N + i for j in range(N)]
        col_list = col_list + index_col_list
        row_list = row_list + [i + N for j in range(N)]
        data_list = data_list + [1 for j in range(N)]

    index_col_list = [j * (N + 1) for j in range(N)]
    col_list = col_list + index_col_list
    row_list = row_list + [2 * N - 1 for j in range(N)]
    data_list = data_list + [1 for j in range(N)]

    row_list_1 = [i for i in range(N * N)]
    row_list_2 = [i + N * N for i in row_list]
    row_list_3 = [i + 2 * N for i in row_list_2]
    row_list = row_list_1 + row_list_2 + row_list_3

    data_list = data_list + [-i for i in data_list]
    col_list = col_list * 2
    data_list = [-1 for i in range(N * N)] + data_list
    col_list = [i for i in range(N * N)] + col_list

    b3 = np.ones(2 * N)
    b3[2 * N - 1] = 0
    b4 = -b3
    b = matrix(b2.tolist() + b3.tolist() + b4.tolist())
    A = spmatrix(data_list, row_list, col_list)

    sol = solvers.lp(D, A, b, solver="glpk")
    P = np.array(sol['x']).reshape(N, N)
    P[P < 10 ** (-3)] = 0
    P[P > 1 - 10 ** (-3)] = 1

    res = data_temp.copy()
    res[:, 1] = np.matmul(P, res[:, 1])

    res = torch.Tensor(res).to(device)
    return res

class RieszArchRDERIE(nn.Module):

    def __init__(self, learner):
        super(RieszArchRDERIE, self).__init__()
        self.learner = learner
        # Scharfstein-Rotnitzky-Robins correction parameter
        self.beta1 = nn.Parameter(torch.zeros(1))
        self.beta2 = nn.Parameter(torch.zeros(1))
        self.beta3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        out = self.learner(x)
        # Scharfstein-Rotnitzky-Robins corrected output
        srr1 = out[:, [0]] + self.beta1 * out[:, [1]]
        srr2 = out[:, [2]] + self.beta2 * out[:, [3]]
        srr3 = out[:, [4]] + self.beta3 * out[:, [5]]
        return torch.cat([out, srr1, srr2, srr3], dim=1)


class RieszNetRDERIE:

    def __init__(self, learner, moment_fn_1, moment_fn_2, moment_fn_3, a_prime=1, a_star=1):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        moment_fn : a function that takes as input a tuple (x, adversary, device) and
            evaluates the moment function at each of the x's, for a test function given by the adversary model.
            The adversary is a torch model that take as input x and return the output of the test function.
        """
        self.learner = RieszArchRDERIE(learner)
        self.moment_fn_1 = moment_fn_1
        self.moment_fn_2 = moment_fn_2
        self.moment_fn_3 = moment_fn_3
        self.a_prime = a_prime
        self.a_star = a_star

    def _pretrain(self, X, y, Xval, yval, Xval_supp, *, bs,
                  warm_start, logger, model_dir, device, verbose):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name
        self.device = device

        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)
        if not torch.is_tensor(y):
            y = torch.Tensor(y).to(self.device)
        if (Xval is not None) and (not torch.is_tensor(Xval)):
            Xval = torch.Tensor(Xval).to(self.device)
        if (yval is not None) and (not torch.is_tensor(yval)):
            yval = torch.Tensor(yval).to(self.device)
        if (Xval_supp is not None) and (not torch.is_tensor(Xval_supp)):
            Xval_supp = torch.Tensor(Xval_supp).to(self.device)
        y = y.reshape(-1, 1)
        yval = yval.reshape(-1, 1)

        self.train_ds = TensorDataset(X, y)
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        self.logger = logger
        if self.logger is not None:
            self.writer = SummaryWriter()

        return X, y, Xval, yval, Xval_supp

    def _train(self, X, y, *, Xval, yval, Xval_supp,
               earlystop_rounds, earlystop_delta,
               learner_l2, learner_l1, learner_lr,
               n_epochs, bs, target_reg_1, riesz_weight_1,
               target_reg_2, riesz_weight_2, target_reg_3,
               riesz_weight_3, optimizer):

        parameters = add_weight_decay(self.learner, learner_l2)
        if optimizer == 'adam':
            self.optimizerD = optim.Adam(parameters, lr=learner_lr)
        elif optimizer == 'rmsprop':
            self.optimizerD = optim.RMSprop(parameters, lr=learner_lr, momentum=.9)
        elif optimizer == 'sgd':
            self.optimizerD = optim.SGD(parameters, lr=learner_lr, momentum=.9, nesterov=True)
        else:
            raise AttributeError("Not implemented")

        reg_fn_1 = lambda x: self.learner(x)[:, [0]]
        riesz_fn_1 = lambda x: self.learner(x)[:, [1]]
        reg_fn_2 = lambda x: self.learner(x)[:, [2]]
        riesz_fn_2 = lambda x: self.learner(x)[:, [3]]
        reg_fn_3 = lambda x: self.learner(x)[:, [4]]
        riesz_fn_3 = lambda x: self.learner(x)[:, [5]]

        if Xval is not None:
            min_eval = np.inf
            time_since_last_improvement = 0
            best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
                                                                patience=5, threshold=0.0, threshold_mode='abs',
                                                                cooldown=0, min_lr=0,
                                                                eps=1e-08, verbose=(self.verbose > 0))

        for epoch in range(n_epochs):

            if self.verbose > 0:
                print("Epoch #", epoch, sep="")

            for it, (xb, yb) in enumerate(self.train_dl):

                self.learner.train()
                output = self.learner(xb)
                if riesz_weight_3 != 0:
                    x_supp_b = permutation_matrix(xb, self.device)
                    output_supp = self.learner(x_supp_b)

                L1_reg_loss = 0.0
                if learner_l1 > 0.0:
                    L1_reg_loss = L1_reg(self.learner, learner_l1)

                # The reg loss function E(Y - theta3)^2
                D_loss = torch.mean((yb - output[:, [4]]) ** 2)

                # The reg loss function E(m3 - theta2)^2
                D_loss += torch.mean((self.moment_fn_3(
                    xb, reg_fn_3, self.device, self.a_prime) - output[:, [2]]) ** 2)

                # The reg loss function E(m2 - theta1)^2
                D_loss += torch.mean((self.moment_fn_2(
                    xb, reg_fn_2, self.device, self.a_star) - output[:, [0]]) ** 2)

                if riesz_weight_1 != 0:
                    # The loss function related to alpha1
                    D_loss += riesz_weight_1 * torch.mean(- 2 * self.moment_fn_1(
                        xb, riesz_fn_1, self.device, self.a_star) + output[:, [1]] ** 2)

                # The TMLE loss related to theta3
                D_loss += target_reg_3 * torch.mean((yb - output[:, [8]]) ** 2)

                if riesz_weight_2 != 0:
                    # The loss function related to alpha2
                    D_loss += riesz_weight_2 * torch.mean(- 2 * self.moment_fn_2(
                        xb, riesz_fn_2, self.device, self.a_star) * output[:, [1]] + output[:, [3]] ** 2)

                if riesz_weight_3 != 0:
                    # The loss function related to alpha3
                    D_loss += riesz_weight_3 * torch.mean(- 2 * self.moment_fn_3(
                        x_supp_b, riesz_fn_3, self.device, self.a_prime) * output_supp[:, [3]] + output[:, [5]] ** 2)

                # The TMLE loss related to theta1
                D_loss += target_reg_1 * torch.mean((self.moment_fn_2(
                    xb, reg_fn_2, self.device, self.a_star) - output[:, [6]]) ** 2)

                # The TMLE loss related to theta2
                D_loss += target_reg_2 * torch.mean((self.moment_fn_3(
                    xb, reg_fn_3, self.device, self.a_prime) - output[:, [7]]) ** 2)

                # The regularization loss
                D_loss += L1_reg_loss

                self.optimizerD.zero_grad()
                D_loss.backward()
                self.optimizerD.step()
                self.learner.eval()

            if Xval is not None:  # if early stopping was enabled we check the out of sample violation
                output = self.learner(Xval)
                output_supp = self.learner(Xval_supp)
                loss1 = np.mean(torch.mean((yval - output[:, [4]]) ** 2).cpu().detach().numpy())  # theta3
                loss2 = np.mean(torch.mean((self.moment_fn_3(
                    Xval, reg_fn_3, self.device, self.a_prime) - output[:, [2]]) ** 2).cpu().detach().numpy())  # theta2
                loss3 = np.mean(torch.mean((self.moment_fn_2(
                    Xval, reg_fn_2, self.device, self.a_star) - output[:, [0]]) ** 2).cpu().detach().numpy())  # theta1

                loss4 = np.mean(torch.mean(- 2 * self.moment_fn_1(
                    Xval, riesz_fn_1, self.device, self.a_star) + output[:, [1]] ** 2).cpu().detach().numpy())  # alpha1

                loss5 = np.mean(torch.mean(- 2 * self.moment_fn_2(
                    Xval, riesz_fn_2, self.device, self.a_star) * output[:, [1]] + output[:, [
                                                                                                 3]] ** 2).cpu().detach().numpy())  # alpha2
                loss6 = np.mean(torch.mean(- 2 * self.moment_fn_3(
                    Xval_supp, riesz_fn_3, self.device, self.a_prime) * output_supp[:, [3]] + output[:, [
                                                                                                            5]] ** 2).cpu().detach().numpy())  # alpha3

                loss7 = np.mean(torch.mean((self.moment_fn_2(
                    Xval, reg_fn_2, self.device, self.a_star) - output[:,
                                                                [6]]) ** 2).cpu().detach().numpy())  # theta1 tmle

                loss8 = np.mean(torch.mean((self.moment_fn_3(
                    Xval, reg_fn_3, self.device, self.a_prime) - output[:,
                                                                 [7]]) ** 2).cpu().detach().numpy())  # theta2 tmle

                loss9 = np.mean(torch.mean((yval - output[:, [8]]) ** 2).cpu().detach().numpy())  # theta3 tmle

                self.curr_eval = (loss1 + loss2 + loss3 + riesz_weight_1 * loss4 +
                                  riesz_weight_2 * loss5 + riesz_weight_3 * loss6 + target_reg_1 * loss7 +
                                  target_reg_2 * loss8 + target_reg_3 * loss9)

                lr_scheduler.step(self.curr_eval)

                if self.verbose > 0:
                    print("Validation losses:", loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9,
                          self.curr_eval)
                if min_eval > self.curr_eval + earlystop_delta:
                    min_eval = self.curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            if self.logger is not None:
                self.logger(self, self.learner, epoch, self.writer)

        torch.save(self.learner, os.path.join(
            self.model_dir, "epoch{}".format(epoch)))

        self.n_epochs = epoch + 1
        if Xval is not None:
            self.learner.load_state_dict(best_learner_state_dict)
            torch.save(self.learner, os.path.join(
                self.model_dir, "earlystop"))

        return self

    def fit(self, X, y, Xval=None, yval=None, Xval_supp=None, *,
            earlystop_rounds=20, earlystop_delta=0,
            learner_l2=1e-3, learner_l1=0, learner_lr=0.001,
            n_epochs=100, bs=100, target_reg_1=.1, riesz_weight_1=1.0, target_reg_2=.1, riesz_weight_2=1.0,
            target_reg_3=.1, riesz_weight_3=1.0, optimizer='adam', warm_start=False,
            logger=None, model_dir='.', device=None, verbose=0):
        """
        Parameters
        ----------
        X : features of shape (n_samples, n_features)
        y : label of shape (n_samples, 1)
        Xval : validation set, if not None, then earlystopping is enabled based on out of sample moment violation
        yval : validation labels
        earlystop_rounds : how many epochs to wait for an out of sample improvement
        earlystop_delta : min increment for improvement for early stopping
        learner_l2 : l2_regularization of parameters of learner
        learner_l1 : l1_regularization of parameters of learner
        learner_lr : learning rate of the Adam optimizer for learner
        n_epochs : how many passes over the data
        bs : batch size
        target_reg : float in [0, 1]. weight on targeted regularization vs mse loss
        optimizer : one of {'adam', 'rmsprop', 'sgd'}. default='adam'
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        device : name of device on which to perform all computation
        verbose : whether to print messages related to progress of training

        Args:
            target_reg_1: ...
            riesz_weight_1: ...
            target_reg_2: ...
            riesz_weight_2: ...
        """

        X, y, Xval, yval, Xval_supp = self._pretrain(X, y, Xval, yval, Xval_supp, bs=bs, warm_start=warm_start,
                                                     logger=logger, model_dir=model_dir,
                                                     device=device, verbose=verbose)

        self._train(X, y, Xval=Xval, yval=yval, Xval_supp=Xval_supp,
                    earlystop_rounds=earlystop_rounds, earlystop_delta=earlystop_delta,
                    learner_l2=learner_l2, learner_l1=learner_l1,
                    learner_lr=learner_lr, n_epochs=n_epochs, bs=bs,
                    target_reg_1=target_reg_1, riesz_weight_1=riesz_weight_1, target_reg_2=target_reg_2,
                    riesz_weight_2=riesz_weight_2, target_reg_3=target_reg_3, riesz_weight_3=riesz_weight_3,
                    optimizer=optimizer)

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def get_model(self, model):
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))
        if model == 'earlystop':
            return torch.load(os.path.join(self.model_dir,
                                           "earlystop"))

        raise AttributeError("Not implemented")

    def predict(self, X, model='final'):
        """
        Parameters
        ----------
        X : (n, p) matrix of features
        model : one of ('final', 'earlystop'), whether to use an average of models or the final
        Returns
        -------
        ypred, apred : (n, 2) matrix of learned regression and riesz representers g(X), a(X)
        """
        if not torch.is_tensor(X):
            X = torch.Tensor(X).to(self.device)

        return self.get_model(model)(X).cpu().data.numpy()

    def predict_avg_moment(self, Xtest, ytest, method='dr',
                           model='final', alpha=0.05, srr=True, postTMLE=False, correctionmethod='residuals',
                           postproc_riesz=False):
        """
        Parameters
        ----------
        Xtest : (n, p) matrix of features
        ytest : (n,) vector of labels
        method : one of ('dr', 'ips', 'reg') for approach
        model : one of ('final', 'earlystop'), whether to use the final or the earlystop model
        alpha : confidence level, creates (1 - alpha)*100% confidence interval
        srr : whether to apply Scharfstein-Rotnitzky-Robins correction to regressor
        Returns
        -------
        avg_moment, lb, ub: avg moment with confidence intervals
        """
        if not torch.is_tensor(Xtest):
            Xtest = torch.Tensor(Xtest).to(self.device)
        if torch.is_tensor(ytest):
            ytest = ytest.cpu().data.numpy()
        ytest = ytest.flatten()

        pred_test = self.predict(Xtest, model=model)
        a_test = pred_test[:, 3]  # This is the alpha_2
        a_test_3 = pred_test[:, 5]  # This is the alpha_3
        a_test_2 = pred_test[:, 3]  # This is the alpha_2
        a_test_1 = pred_test[:, 1]  # This is the alpha_1
        # Robins-Rotnitzky-Scharfstein correction or not
        y_pred_test = pred_test[:, 8] if srr and not postTMLE else pred_test[:, 4]  # This is the theta_3
        y_pred_test_3 = pred_test[:, 8] if srr and not postTMLE else pred_test[:, 4]  # This is the theta_3
        y_pred_test_2 = pred_test[:, 7] if srr and not postTMLE else pred_test[:, 2]  # This is the theta_2
        y_pred_test_1 = pred_test[:, 6] if srr and not postTMLE else pred_test[:, 0]  # This is the theta_1
        agmm_model = self.get_model(model)
        reg_fn_1 = lambda x: agmm_model(x)[:, [6]] if srr and not postTMLE else agmm_model(x)[:, [0]]
        reg_fn_2 = lambda x: agmm_model(x)[:, [7]] if srr and not postTMLE else agmm_model(x)[:, [2]]
        reg_fn_3 = lambda x: agmm_model(x)[:, [8]] if srr and not postTMLE else agmm_model(x)[:, [4]]
        riesz_fn = lambda x: agmm_model(x)[:, [1]]

        adj_reg_fn = reg_fn_2

        if method == 'direct':
            return mean_ci(
                self.moment_fn_2(Xtest, adj_reg_fn, self.device, self.a_star).cpu().detach().numpy().flatten(),
                confidence=1 - alpha)
        elif method == 'ips':
            return mean_ci(a_test * ytest, confidence=1 - alpha)
        elif method == 'dr':
            # This is the m2, which should be the true outcome for theta1
            # true_outcome = self.moment_fn_2(Xtest, reg_fn_2, self.device).cpu().detach().numpy().flatten()

            # return mean_ci((self.moment_fn_2(Xtest, adj_reg_fn, self.device).cpu().detach().numpy().flatten()
            #                 + a_test * (true_outcome - y_pred_test)),
            #                confidence=1 - alpha)

            return mean_ci(
                (self.moment_fn_2(Xtest, adj_reg_fn, self.device, self.a_star).cpu().detach().numpy().flatten()
                 + a_test * (ytest - y_pred_test)),
                confidence=1 - alpha)

        elif method == 'test':
            m_1 = self.moment_fn_1(Xtest, reg_fn_1, self.device, self.a_star).cpu().detach().numpy().flatten()
            m_2 = self.moment_fn_2(Xtest, reg_fn_2, self.device, self.a_star).cpu().detach().numpy().flatten()
            m_3 = self.moment_fn_3(Xtest, reg_fn_3, self.device, self.a_prime).cpu().detach().numpy().flatten()
            return mean_ci((m_1
                            + a_test_3 * (ytest - y_pred_test)) + a_test_2 * (m_3 - m_2) + a_test_1 * (m_2 - m_1),
                           confidence=1 - alpha)

        else:
            raise AttributeError('not implemented')
