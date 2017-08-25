from __future__ import division
from matplotlib import colors as mcolors
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from numpy.linalg import svd, norm, matrix_rank, pinv, inv
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# import fancyimpute simple impute


def MAR(X, p):
    pass


# ensure treatment unit is 'zeroth' unit
def swap(X, unit):
    X[[0, unit], :] = X[[unit, 0], :]
    return X


# singular value thresholding
def threshold(X, num_sv=1):
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
    D = np.zeros((m, n))
    D[:S_size, :S_size] = np.diag(S)
    M_hat = (1 / p_hat) * np.dot(U, np.dot(D, V))

    # convert matrix back to original dimensions
    if transform == 1:
        M_hat = M_hat.T

    return np.real(M_hat)


# PRINCIPLE COMPONENT REGRESSION
def PCR(A, y, regr, alphas, cv):
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('regr', regr)])

    # Prediction
    donor_size = A.shape[1]
    n_components = np.linspace(1, donor_size, donor_size)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs), cv=cv)
    estimator.fit(X_digits, y_digits)


# compute unnormalized mse
def mse(X, y, beta):
    y_hat = X.dot(beta)
    return norm(y_hat - y) ** 2


# forward chaining method: cross-validation for time series
# to maintain causal structure of data
def forward_chain(X, y, method="Ridge"):
    # forward chaining strategy
    N = 100
    lmda = np.linspace(0.1, 30, N)
    penalties = np.zeros(len(lmda))
    year = X.shape[0]

    for i in range(len(lmda)):
        penalty = 0
        for t in range(1, year):
            # train_test_split
            X_train = X[:t, :]
            y_train = y[:t]
            X_test = X[t, :]
            y_test = y[t]

            # fit model
            if method == "Lasso":
                regr = linear_model.Lasso(lmda[i], fit_intercept=False)
            else:
                regr = linear_model.Ridge(lmda[i], fit_intercept=False)
            regr.fit(X_train, y_train)
            beta = regr.coef_

            # temporary score
            penalty += mse(X_test, y_test, beta)
        penalties[i] = penalty / year

    return lmda[np.argmin(penalties)]


# inference stage
def learn(X, year, num_sv=1, method="Linear"):
    # filter out noise (threshold data matrix)
    M_hat = threshold(X[1:, :], num_sv=num_sv)
    y = X[0, :year]
    A = M_hat[:, :year].T
    sigma_hat = []

    if method == "Ridge":
        lmda_hat = forward_chain(A, y, method)
        regr = linear_model.Ridge(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        beta = regr.coef_

    elif method == "Lasso":
        lmda_hat = forward_chain(A, y, method)
        regr = linear_model.Lasso(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        beta = regr.coef_

    elif method == "Bayes":
        print("Bayesian Method")
        # Posterior distribution parameters
        inv_var = 1 / np.var(y)
        # prior_param = lmda_hat * inv_var
        prior_param = 0.09
        donor_size = A.shape[1]

        # covariance matrix
        sigma_d = inv(prior_param * np.eye(donor_size) + inv_var * A.T.dot(A)

        # mean vector
        beta=inv_var * np.dot(sigma_d, np.dot(A.T, y))

        # predict posterior variance
        sigma_hat=np.ones(M_hat.shape[1]) / inv_var
        for i in range(M_hat.shape[1]):
            sigma_hat[i] += np.dot(M_hat[:, i].T, np.dot(sigma_d, M_hat[:, i]))
        sigma_hat=np.sqrt(sigma_hat)

    else:
        regr=linear_model.LinearRegression(fit_intercept=False)
        regr.fit(A, y)
        beta=regr.coef_

    # predict counterfactual
    m1=A.dot(beta)
    m2=M_hat[:, year:].T.dot(beta)

    return beta, np.concatenate([m1, m2]), sigma_hat


""" SYNTH CLASS """


class Synth():
    def __init__(self, treat_unit, year, method="Linear", p=1):
        self.treat_unit=treat_unit
        self.year=year
        self.p=p
        self.method=method
        self.beta=[]
        self.mean=[]
        self.orig=[]

    def fit(self, df, num_sv=1, drop=False, drop_list=[]):
        data=df.copy()
        self.num_sv=num_sv
        X=data.as_matrix()
        donor_list=list(data.index)

        # treated unit
        unit=donor_list.index(self.treat_unit)

        # let row zero represent the treatment unit
        X=swap(X, unit)
        self.orig=np.copy(X[0, :])
        self.Y=np.copy(X)

        # estimation
        self.beta, self.mean, self.sigma_hat=learn(
            X, self.year, num_sv=self.num_sv, method=self.method)

    def vis_data(self, xlabel="year", ylabel="metric", title="Case Study",
                 orig_label="observed data", mean_label="counterfactual mean",
                 year_shift=0, year_mod=5, loc="best",
                 lw=1.75, frame_color='0.925'):
        self.xlabel=xlabel
        self.ylabel=ylabel
        self.title=title
        self.orig_label=orig_label
        self.mean_label=mean_label
        self.year_shift=year_shift
        self.year_mod=year_mod
        self.loc=loc
        self.lw=lw
        self.frame_color=frame_color

    def vis(self):
        fig, ax=plt.subplots()
        ax.plot(self.orig, label=self.orig_label, linewidth=self.lw, color='g')
        ax.plot(self.mean, '--', label=self.mean_label, linewidth=self.lw, color='b')
        x_=np.linspace(0, len(self.mean) - 1, len(self.mean))
        clr1='lightcyan'
        clr2='paleturquoise'
        upper=self.mean + self.sigma_hat
        lower=self.mean - self.sigma_hat
        ax.fill_between(x_, self.mean, upper, facecolor=clr1, edgecolor=clr2, interpolate=True)
        ax.fill_between(x_, self.mean, lower, facecolor=clr1, edgecolor=clr2, interpolate=True)
        legend=ax.legend(loc=self.loc, shadow=True, prop={'size': 9.5})
        frame=legend.get_frame()
        frame.set_facecolor(self.frame_color)
        ax.plot([self.year, self.year], [ax.get_ylim()[0], ax.get_ylim()[1]],
                '--', linewidth=self.lw, color='r')
        years=int(np.floor(self.orig.shape[0] / self.year_mod))
        x=np.array([self.year_mod * i for i in range(years + 1)])
        ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1]])
        plt.xticks(x, x + self.year_shift)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.show()
        plt.close()
