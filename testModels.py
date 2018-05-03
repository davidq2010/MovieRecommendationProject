# Models to import
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Data manipulation
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from getDataFrames import getXYdataFrames

# Command line
import sys, getopt

def __splitData(_X, _Y):
    X_train, X_test, y_train, y_test = train_test_split(_X, _Y, test_size = 0.2)
    print("Training Dimensions: ", X_train.shape, y_train.shape)
    print("Testing Dimensions: ", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def testModel(_model, _X, _Y):
    if _model == "LogisticRegression":
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    elif _model == "MLPClassifier":
        model = MLPClassifier(solver = 'lbfgs')

    X_train, X_test, y_train, y_test = __splitData(_X, _Y)

    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    print("Accuracy of ", _model, " classifier on test set: {:.2f}".format(
        model.score(X_test, y_test)))

def main(argv):
    """
    Takes the command line arguments (after the name of the file) and uses them
    to get the X and Y dataFrames
    """
    errorString = "Usage: testModels.py <featureFile> <targetFile>"
    if len(argv) != 2:
        print(errorString)
        sys.exit(2)
    featureFile = ''
    targetFile = ''
    try:
        opts, args = getopt.getopt(argv, "h")
    except getOpt.GetoptError:
        print(errorString)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(errorString)
            sys.exit()
    featureFile = argv[0]
    targetFile = argv[1]
    print("Feature file: ", featureFile)
    print("Target file: " , targetFile)
    X,Y = getXYdataFrames(featureFile, targetFile)
    testModel("LogisticRegression", X, Y)

if __name__ == "__main__":
    main(sys.argv[1:])
