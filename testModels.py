# Models to import
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Data manipulation
from matplotlib import pyplot as plt
from DataPreprocessing import getData, preprocessData

# Command line
import sys, getopt

def testModel(_model, X_train, X_test, y_train, y_test):
    """
    X_train and X_test should be preprocessed.
    """
    if _model == "LogisticRegression":
        model = LogisticRegression(solver='sag')
    elif _model == "MLPClassifier":
        model = MLPClassifier()
    elif _model == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif _model == "GradientBoostingClassifier":
        model = GradientBoostingClassifier()
    elif _model == "XGBClassifier":
        model = XGBClassifier()

    # Since XGBoost is not part of sklearn
    if _model == "XGBClassifier":
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy of ", _model, " classifier on test set: {:.2f}".format(
            accuracy))

    # For the sklearn stuff
    else:
        #selector = RFE(model, 5)
        selector = RFECV(model)    # Use the RFE wrapper
        selector.fit(X_train, y_train.values.ravel())
        y_pred = selector.predict(X_test)
        print("Accuracy of ", _model, " classifier on test set: {:.2f}".format(
            selector.score(X_test, y_test)))

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
    X_train, X_test, y_train, y_test = getData(featureFile, targetFile)
    columns = ['Gender', 'Age', 'Occupation','Region']
    X_train, X_test = preprocessData(X_train, X_test, columns)
    #testModel("LogisticRegression", X_train, X_test, y_train, y_test)
    testModel("RandomForestClassifier", X_train, X_test, y_train, y_test)
    #testModel("MLPClassifier", X_train, X_test, y_train, y_test)
    #testModel("GradientBoostingClassifier", X_train, X_test, y_train, y_test)
    #testModel("XGBClassifier",X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main(sys.argv[1:])
