import numpy as np
from numpy.linalg import solve
import scipy as sp

import sklearn
import sklearn.datasets
import sklearn.linear_model

###########################################################
# load the boston housing dataset

boston = sklearn.datasets.load_boston()
X = boston.data
y = boston.target

###########################################################
# fit a regression model using matrix algebra,
# and without using sklearn other than to grab
# the data, using all the variables of X to
# predict the response y; don't forget the
# intercept; try to get both the regression
# vector beta and the predicted values y-hat

# add ones to the data matrix
X = np.concatenate([np.ones([X.shape[0], 1]), X], axis = 1)

# compute matrix values and find regression vector
Xt = np.transpose(X)
XtX = np.dot(Xt, X)
Xty = np.dot(Xt, y)
beta = solve(XtX, Xty)

# compute fitted values
yhat = np.dot(X, beta)

###########################################################
# now, repeat this using sklearn
X = boston.data
y = boston.target

lr = sklearn.linear_model.LinearRegression()
lr.fit(X, y)

beta_sklearn = np.concatenate([[lr.intercept_], lr.coef_])
yhat_sklearn = lr.predict(X)

# compare them
np.max(np.abs(beta - beta_sklearn))
np.max(np.abs(yhat - yhat_sklearn))


