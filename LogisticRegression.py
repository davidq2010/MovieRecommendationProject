"""
This class implements Logistic Regression 
Reference: https://beckernick.github.io/logistic-regression-from-scratch/

"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

print(simulated_separableish_features)
print(len(simulated_separableish_features))
print(len(simulated_labels))
plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)

# @brief: Transform a linear model of the predictors 
# by bounding the target output between 0 and 1
# @param scores: a vector of target output
# @return a transformed vector of target output bounded between 0 and 1
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


# def log_likelihood(features, target, weights):
#     scores = np.dot(features, weights)
#     ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
#     return ll

# @brief: Calculate the weights for the logistic regression model
# based on training data.
# @param features: the given features data (X)
# @param target: the given expected output (Y)
# @param num_steps: the given step to finesse the weights calculated for
#		 a higher accuracy rate
# @param learning_rate: the base size of a "step" in the gradient descent
# @param add_intercept: true if the model include B0 interept, false otherwise
# @return a vector of weights of the model 
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
    	# create a vector filled with 1s.
        intercept = np.ones((features.shape[0], 1))
        # concatenate features to intercept
        features = np.hstack((intercept, features))
        
    # create a vector filled with zeros    
    weights = np.zeros(features.shape[1])
    
    # Keep increase the likelihood of the weights fitting the model
    # by adapting the weights each step
    for step in xrange(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient,
        # gradient is proportional to the error 
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
        # # Print log-likelihood every so often
        # if step % 10000 == 0:
        #     print log_likelihood(features, target, weights)
        
    return weights

# Calculate the weights using the function we implemented
weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 300000, learning_rate = 5e-5, add_intercept=True)

# Compare accuracy with skLearn Logistic regression 
from sklearn.linear_model import LogisticRegression

# C is inverse of regularization strength
# High C translates to weaker regularization, which means higher chances of overfitting
# to correctly categorize data points
clf = LogisticRegression(fit_intercept=True, C = 1)
clf.fit(simulated_separableish_features, simulated_labels)

print clf.intercept_, clf.coef_
print weights

data_with_intercept = np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features))
final_scores = np.dot(data_with_intercept, weights)
preds = np.round(sigmoid(final_scores))

print 'Accuracy from scratch: {0}'.format((preds == simulated_labels).sum().astype(float) / len(preds))
print 'Accuracy from sk-learn: {0}'.format(clf.score(simulated_separableish_features, simulated_labels))

plt.figure(figsize = (12, 8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = preds == simulated_labels - 1, alpha = .8, s = 50)