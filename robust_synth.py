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


def visuals(estimate, raw, estimate_label="", raw_label="", year=0, year_shift=0, xlabel="", ylabel="", title="",
            legend_loc='upper left', year_mod=5, frame_color='0.925', line_width=1.75):
    # Plot data
    fig, ax = plt.subplots()
    ax.plot(raw[:], label=raw_label, lw=line_width, color='k')
    ax.plot(estimate[:], '--', label=estimate_label, lw=line_width, color='b')
    legend = ax.legend(loc=legend_loc, shadow=True, prop={'size': 10.5})
    frame = legend.get_frame()
    frame.set_facecolor(frame_color)
    ax.plot([year, year], [ax.get_ylim()[0], ax.get_ylim()[1]], '--', linewidth=1.5, color='r')
    years = int(np.floor(raw.shape[0] / year_mod))
    x = np.array([year_mod * i for i in range(years + 1)])
    plt.xticks(x, x + year_shift)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.rcParams.update({'font.size': 10})
    plt.show()


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
def score(X, y, beta):
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
            penalty += score(X_test, y_test, beta)
        penalties[i] = penalty / year

    lmda_hat = lmda[np.argmin(penalties)]


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
        w = regr.coef_

    elif method == "Lasso":
        lmda_hat = forward_chain(A, y, method)
        regr = linear_model.Lasso(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        w = regr.coef_

    elif method == "Bayes":
        print("Bayesian Method")
        # Posterior distribution parameters
        inv_var = 1 / np.var(y)
        #prior_param = lmda_hat * inv_var
        prior_param = 0.09
        donor_size = A.shape[1]

        # covariance matrix
        sigma_d = inv(prior_param * np.eye(donor_size) + inv_var * np.dot(A.T, A))

        # mean vector
        w = inv_var * np.dot(sigma_d, np.dot(A.T, y))

        # predict posterior variance
        sigma_hat = np.ones(M_hat.shape[1]) / inv_var
        for i in range(M_hat.shape[1]):
            sigma_hat[i] += np.dot(M_hat[:, i].T, np.dot(sigma_d, M_hat[:, i]))
        sigma_hat = np.sqrt(sigma_hat)

    else:
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(A, y)
        w = regr.coef_

    # predict counterfactual
    m1 = A.dot(w)
    m2 = M_hat[:, year:].T.dot(w)

    return w, np.concatenate([m1, m2]), sigma_hat


""" SYNTH CLASS """


class Synth():
    def __init__(self, treat_unit, year, method="Linear", p=1):
        self.treat_unit = treat_unit
        self.year = year
        self.p = p
        self.method = method
        self.w = []
        self.estimate = []
        self.raw = []

    def fit(self, df, num_sv=1, drop=False, drop_list=[]):
        data = df.copy()
        self.num_sv = num_sv

        # prepare data
        self.drop = drop
        if self.drop:
            data = data.drop(data.index[drop_list])
        X = data.as_matrix()
        donor_list = list(data.index)

        # treated unit
        unit = donor_list.index(self.treat_unit)

        # let row zero represent the treatment unit
        X = swap(X, unit)
        self.raw = np.copy(X[0, :])
        self.Y = np.copy(X)

        # estimation
        self.w, self.estimate, self.sigma_hat = learn(
            X, self.year, num_sv=self.num_sv, method=self.method)

    def vis_data(self, xlabel="year", ylabel="", title="", year_shift=0, year_mod=5,
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
        fig, ax = plt.subplots()
        ax.plot(self.raw[:], label="orig", lw=1.5, color='b')
        ax.plot(self.estimate[:], '--', label="counterfactual", lw=1.5, color='g')
        legend = ax.legend(loc="upper left", shadow=True, prop={'size': 10.5})
        plt.show()

        """self.estimate_label = "Synthetic " + self.treat_unit
        self.raw_label = self.treat_unit
        if self.drop:
            visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                    self.xlabel, self.ylabel, self.title + " - no bad states", legend_loc=self.legend_loc,
                    year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)
        else:
            visuals(self.estimate, self.raw, self.estimate_label, self.raw_label, self.year, self.year_shift,
                    self.xlabel, self.ylabel, self.title, legend_loc=self.legend_loc,
                    year_mod=self.year_mod, line_width=self.line_width, frame_color=self.frame_color)
                    """
