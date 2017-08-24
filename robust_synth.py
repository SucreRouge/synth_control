import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from numpy.linalg import svd, norm, matrix_rank, pinv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from __future__ import division
from matplotlib import colors as mcolors


def MAR(X, p):
    # missing at random
    # p = proportion of observed entries
    m, n = X.shape
    mn = m * n
    k = int(p * mn)
    Y = np.random.choice([0, 1], size=mn, p=[1 - p, p])
    return np.where(np.reshape(Y, (m, n)) == 0), np.where(np.reshape(Y, (m, n)) == 1)


def impute(Y, idx, idx2, method="avg"):
    # impute missing entries
    X = np.copy(Y)
    X = X.astype(np.float)
    X[idx] = np.median(X)
    return X


def swap(X, unit):
    X[[0, unit], :] = X[[unit, 0], :]
    return X


def threshold(X, num_sv=1, eta=0):
    # enforce data matrix X (m x n) to be a fat matrix (m <= n)
    transform = 0
    if X.shape[0] > X.shape[1]:
        X = X.T
        transform = 1
    m, n = X.shape

    # proportion of observed entries
    p_hat = np.count_nonzero(X) / (m * n)

    # transform data matrix
    Y = np.copy(X)
    Y[np.isnan(Y)] = 0

    # find threshold singular values
    U, s, V = np.linalg.svd(Y, full_matrices=True)
    S = s[:num_sv]
    S_size = len(S)

    # create matrix W
    D = np.zeros((m, n), dtype=complex)
    D[:S_size, :S_size] = np.diag(S)
    M_hat = (1 / p_hat) * np.dot(U, np.dot(D, V))

    # convert matrix back to original dimensions
    if transform == 1:
        M_hat = M_hat.T

    return np.real(M_hat)


def learning(X, year, num_sv=1, method="linear", eta=0):
    # filter out noise (threshold data matrix)
    M_hat = threshold(X[1:, :], num_sv=num_sv, eta=0)
    y = X[0, :year]
    A = M_hat[:, :year].T

    # cross-validation (LOO)
    N = 100
    lmda = np.linspace(0.1, 30, N)
    penalties = np.zeros(len(lmda))

    """# cross validation (LOO)
    for i in range(len(lmda)):
        penalty = 0
        for t in range(year):
            # copy data
            Y = np.copy(y)
            X = np.copy(A)

            # LOO
            y_test = Y[t]
            X_test = X[t, :]

            X_train = np.delete(X, (t), axis=0)
            y_train = np.delete(Y, t)

            # fit model
            regr = linear_model.Ridge(lmda[i], fit_intercept=False)
            regr.fit(X_train, y_train)
            w = regr.coef_

            # temporary score
            y_hat = np.dot(X_test, w)
            temp = np.power(y_test - y_hat, 2)
            penalty += temp
        penalties[i] = penalty / year"""

    # forward chaining strategy
    for i in range(len(lmda)):
        penalty = 0
        for t in range(1, year):
            # copy data
            Y = np.copy(y)
            X = np.copy(A)

            # LOO
            y_test = Y[t]
            X_test = X[t, :]

            X_train = X[:t, :]
            y_train = Y[:t]

            # fit model
            regr = linear_model.Lasso(lmda[i], fit_intercept=False)
            regr.fit(X_train, y_train)
            w = regr.coef_

            # temporary score
            y_hat = np.dot(X_test, w)
            temp = np.power(y_test - y_hat, 2)
            penalty += temp
        penalties[i] = penalty / year

    lmda_hat = lmda[np.argmin(penalties)]
    print(lmda_hat)

    if method == "Ridge":
        print("Ridge Regression")

        # estimate synthetic control
        regr = linear_model.Ridge(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        w = regr.coef_
        sigma_hat = []

    elif method == "Lasso":
        print("LASSO Regression")

        # estimate synthetic control
        regr = linear_model.Lasso(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        w = regr.coef_
        sigma_hat = []

    elif method == "Bayes":
        print("Bayesian Method")

        # Posterior distribution parameters
        inv_var = 1 / np.var(y)
        prior_param = lmda_hat * inv_var
        prior_param = 0.09

        print(prior_param)

        donor_size = A.shape[1]

        # covariance matrix
        sigma_d = np.linalg.inv(prior_param * np.eye(donor_size) + inv_var * np.dot(A.T, A))

        # mean vector
        w = inv_var * np.dot(sigma_d, np.dot(A.T, y))

        # predict posterior variance
        sigma_hat = np.ones(M_hat.shape[1]) / inv_var
        for i in range(M_hat.shape[1]):
            sigma_hat[i] += np.dot(M_hat[:, i].T, np.dot(sigma_d, M_hat[:, i]))
        sigma_hat = np.sqrt(sigma_hat)

    else:
        print("Linear Regression")
        # estimate synthetic control
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(A, y)
        w = regr.coef_
        sigma_hat = []

    # predict counterfactual
    m1 = A.dot(w)
    m2 = M_hat[:, year:].T.dot(w)

    return w, np.concatenate([m1, m2]), M_hat, sigma_hat

    class Synth():
        def __init__(self, treat_unit, year, eta=0, p=1):
            self.treat_unit = treat_unit
            self.year = year
            self.eta = eta
            self.p = p
            self.w = []
            self.estimate = []
            self.raw = []

        def fit(self, df, num_sv=1, method="linear", method2="row avg", drop=0, drop_list=[]):
            data = df.copy()
            self.num_sv = num_sv

            # prepare data
            self.drop = drop
            if self.drop == 1:
                data = data.drop(data.index[drop_list])
            X = data.as_matrix()
            donor_list = list(data.index)

            # treated unit
            unit = donor_list.index(self.treat_unit)

            # let row one be treated unit
            X = swap(X, unit)
            self.raw = np.copy(X[0, :])
            self.Y = np.copy(X)

            # drop p proportion of entries
            idx1, idx2 = MAR(X, self.p)
            X = impute(X, idx1, idx2, method=method2)

            # estimation
            self.method = method
            self.w, self.estimate, self.M_hat, self.sigma_hat = learning(
                X, self.year, num_sv=self.num_sv, method=self.method, eta=self.eta)

        def vis_data(self, xlabel, ylabel, title, year_shift, year_mod=5,
                     legend_loc="upper left", line_width=2.0, frame_color='0.925'):
            self.xlabel = xlabel
            self.ylabel = ylabel
            self.title = title
            self.year_shift = year_shift
            self.year_mod = year_mod
            self.legend_loc = legend_loc
            self.line_width = line_width
            self.frame_color = frame_color

        def vis(self):
            self.estimate_label = "Synthetic " + self.treat_unit
            self.raw_label = self.treat_unit
            if self.drop == 1:
                visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                        self.xlabel, self.ylabel, self.title + " - no bad states", legend_loc=self.legend_loc,
                        year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)
            else:
                visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                        self.xlabel, self.ylabel, self.title, legend_loc=self.legend_loc,
                        year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)
