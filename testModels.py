from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

X = readDataFromFeatureFile('x_train.txt')
Y = readDataFromTargetFile('y_train.txt')

#print(X)
#print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

logReg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
logReg.fit(x_train, y_train.values.ravel())
y_pred = logReg.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logReg.score(x_test, y_test)))
