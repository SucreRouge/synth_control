import numpy as np
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from numpy.linalg import svd, norm, inv

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


# singular value thresholding
def threshold(X, num_sv=1):
    # enforce data matrix X (m x n) to be a fat matrix (m <= n)
    transform = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transform = True
    m, n = X.shape

    # proportion of observed entries
    p_hat = np.count_nonzero(X) / (m * n)

    # transform data matrix
    Y = np.copy(X)
    a = np.nanmin(Y)
    b = np.nanmax(Y)
    Y -= (a + b) / 2
    Y /= (b - a) / 2
    Y[np.isnan(Y)] = 0
    # Y[np.isnan(Y)] = np.nanmedian(X)

    # find threshold singular values
    U, s, V = np.linalg.svd(Y, full_matrices=True)
    S = s[:num_sv]
    S_size = len(S)

    # create matrix W
    D = np.zeros((m, n))
    D[:S_size, :S_size] = np.diag(S)
    M_hat = (1 / p_hat) * np.dot(U, np.dot(D, V))

    # re-transform matrix
    M_hat *= (b - a) / 2
    M_hat += (a + b) / 2

    # convert matrix back to original dimensions
    if transform:
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
    sigma_hat = 0

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
