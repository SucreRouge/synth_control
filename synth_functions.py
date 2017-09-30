from __future__ import division
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from numpy.linalg import svd, norm, inv
from scipy import stats


def t_test(array1, array2):
    return stats.ttest_rel(array1, array2)


# Missing at random (MIGHT NEED TO FIX THIS!!!)
def MAR(X, p):
    missing_mask = np.random.rand(*X.shape) < (1 - p)
    X_incomplete = X.copy()
    X_incomplete[missing_mask] = np.nan
    return X_incomplete


# ensure treatment unit is 'zeroth' unit
def swap(X, unit):
    X[[0, unit], :] = X[[unit, 0], :]
    return X


def transform(X):
    a = np.nanmin(X)
    b = np.nanmax(X)
    X -= (a + b) / 2
    X /= (b - a) / 2
    X[np.isnan(X)] = 0
    return X, a, b


def inverse_transform(X, a, b):
    X *= (b - a) / 2
    X += (a + b) / 2
    return X


# singular value thresholding
def threshold(X, num_sv=1):
    # enforce data matrix X (m x n) to be a fat matrix (m <= n)
    transpose = False
    transform_ = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    m, n = X.shape

    # proportion of observed entries
    p_hat = np.count_nonzero(X) / (m * n)

    # transform data matrix
    Y = np.copy(X)
    if np.nanmin(Y) < -1 or np.nanmax(Y) > 1:
        Y, a, b = transform(Y)
        transform_ = True
    else:
        #Y[np.isnan(Y)] = 0
        Y[np.isnan(Y)] = np.nanmedian(X)

    # find threshold singular values
    U, s, V = np.linalg.svd(Y, full_matrices=True)
    S = s[:num_sv]
    S_size = len(S)

    # create matrix W
    D = np.zeros((m, n))
    D[:S_size, :S_size] = np.diag(S)
    M_hat = (1 / p_hat) * np.dot(U, np.dot(D, V))

    # re-transform matrix
    if transform_:
        M_hat = inverse_transform(M_hat, a, b)

    # convert matrix back to original dimensions
    if transpose:
        M_hat = M_hat.T

    return np.real(M_hat)


# PRINCIPLE COMPONENT REGRESSION
def PCR(A, y, alphas=np.linspace(0.1, 10, 100), cv=1, method="Linear"):
    if method == "Ridge":
        regr = linear_model.Ridge(fit_intercept=False)
    elif method == "Lasso":
        regr = linear_model.Lasso(fit_intercept=False)
    else:
        regr = linear_model.LinearRegression(fit_intercept=False)

    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('regr', regr)])

    # dumb method of PCR: (1) PCA (2) Regression
    # step 1:
    pca = PCA()
    pca.fit(A)

    # Prediction
    donor_size = A.shape[1]
    n_components = np.linspace(1, donor_size, donor_size)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  alphas=alphas), cv=cv)
    estimator.fit(X_digits, y_digits)


# compute unnormalized mse
def mse(X, y, beta):
    y_hat = X.dot(beta)
    return norm(y_hat - y) ** 2


# forward chaining method: cross-validation for time series
# to maintain causal structure of data
def forward_chain(X, y, method="ridge"):
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
            if method.lower() == "lasso":
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
def learn(X, year, num_sv=1, method="linear"):
    # filter out noise (threshold data matrix)
    M_hat = threshold(X[1:, :], num_sv=num_sv)
    y = X[0, :year]
    A = M_hat[:, :year].T
    sigma_hat = 0

    if method.lower() == "ridge":
        lmda_hat = forward_chain(A, y, method)
        regr = linear_model.Ridge(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        beta = regr.coef_

    elif method.lower() == "lasso":
        lmda_hat = forward_chain(A, y, method)
        regr = linear_model.Lasso(lmda_hat, fit_intercept=False)
        regr.fit(A, y)
        beta = regr.coef_

    elif method.lower() == "bayes":
        print("Bayesian Method")
        # Posterior distribution parameters
        inv_var = 1 / np.var(y)

        #regr = linear_model.RidgeCV(fit_intercept=False)
        #regr.fit(A, y)
        #lmda_hat = regr.alpha_
        lmda_hat = forward_chain(A, y, "ridge")
        prior_param = lmda_hat * inv_var

        #prior_param = 0.09
        donor_size = A.shape[1]

        # covariance matrix
        sigma_d = inv(prior_param * np.eye(donor_size) + inv_var * A.T.dot(A))

        # mean vector
        beta = inv_var * np.dot(sigma_d, np.dot(A.T, y))

        # predict posterior variance
        sigma_hat = np.ones(M_hat.shape[1]) / inv_var
        for i in range(M_hat.shape[1]):
            sigma_hat[i] += np.dot(M_hat[:, i].T, np.dot(sigma_d, M_hat[:, i]))
        sigma_hat = np.sqrt(sigma_hat)

    else:
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(A, y)
        beta = regr.coef_

    # predict counterfactual
    m1 = A.dot(beta)
    m2 = M_hat[:, year:].T.dot(beta)

    return beta, np.concatenate([m1, m2]), sigma_hat


# FOR PLOTTING PURPOSES ONLY

def case_study(data, region, year, num_sv=2, method="Linear", abadie=[]):
    case = robust_synth.Synth(region, year=year, method=method, num_sv=num_sv)
    case.fit(data)
    case.vis_data()
    case.vis(abadie)


def synth_plots(obs, linear, ridge, lasso, abadie, title, xlabel, ylabel, region, year, year_shift, loc, upper):
    fig, ax = plt.subplots()
    ax.plot(obs, label=region, linewidth=1.75, color='k')
    ax.plot(linear, '--', label='Synthetic ' + region + " (linear)", linewidth=1.75, color='b')
    ax.plot(ridge, '--', label='Synthetic ' + region + " (ridge)", linewidth=1.75, color='g')
    ax.plot(lasso, '--', label='Synthetic ' + region +
            " (lasso)", linewidth=1.75, color='darkorange')
    ax.plot(abadie, '--', label='Synthetic ' + region +
            " (Abadie et. al)", linewidth=1.75, color='gray')
    legend = ax.legend(loc=loc, shadow=True, prop={'size': 9.5})
    frame = legend.get_frame()
    frame.set_facecolor('0.925')
    ax.plot([year, year], [0, upper], '--', linewidth=1.5, color='r')
    years = int(np.floor(obs.shape[0] / 5))
    x = np.array([5 * i for i in range(years + 1)])
    ax.set_ylim([0, upper])
    plt.xticks(x, x + year_shift)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()
