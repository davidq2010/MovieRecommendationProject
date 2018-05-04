from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
import pandas as pd
import sys, getopt

def main(argv):
    """
    Takes the command line arguments (after the name of the file) and uses them
    to get the X and Y dataFrames
    """
    errorString = "Usage: getDataFrames.py <featureFile> <targetFile>"
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
    print(X_train)
    print(X_test)

def preprocessData(X_train, X_test, columns):
  X_train, X_test = encodeCategoricalFeatures(X_train, X_test, columns)
  X_train, X_test = scaleData(X_train, X_test)
  return X_train, X_test

def encodeCategoricalFeatures(X_train, X_test, columns):
  '''
  Column is a list of categorical features that we want to encode using
  OneHotEncoder
  '''
  le=LabelEncoder()
  # Iterating over all the common columns in train and test
  for col in X_test.columns.values:
    # Encoding only categorical variables
    if X_test[col].dtypes=='object':
      # Using whole data to form an exhaustive list of levels
      data=X_train[col].append(X_test[col])
      le.fit(data.values)
      X_train[col]=le.transform(X_train[col])
      X_test[col]=le.transform(X_test[col])

  enc = OneHotEncoder(sparse = False)
  X_train_1 = X_train
  X_test_1 = X_test
  for col in columns:
    # creating an exhaustive list of all possible categorical values
    data=X_train[[col]].append(X_test[[col]])
    enc.fit(data)
      
    # Fitting One Hot Encoding on train data
    temp = enc.transform(X_train[[col]])
    # Changing the encoded features into a data frame with new column names
    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
      .value_counts().index])
    # In side by side concatenation index values should be same
    # Setting the index values similar to the X_train data frame
    temp=temp.set_index(X_train.index.values)
    # adding the new One Hot Encoded varibales to the train data frame
    X_train_1=pd.concat([X_train_1,temp],axis=1)
    # fitting One Hot Encoding on test data
    temp = enc.transform(X_test[[col]])
    # changing it into data frame and adding column names
    temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
      .value_counts().index])
    # Setting the index for proper concatenation
    temp=temp.set_index(X_test.index.values)
    # adding the new One Hot Encoded varibales to test data frame
    X_test_1=pd.concat([X_test_1,temp],axis=1)

  return X_train_1, X_test_1

def scaleData(X_train, X_test):
  return scale(X_train), scale(X_test)

def getData(_featureFile, _targetFile):
    """
    Retrieves X and Y datasets from csv files and split them into training, and
    test sets
    """
    X = pd.read_csv(_featureFile)
    Y = pd.read_csv(_targetFile)
    X_train, X_test, y_train, y_test = __splitData(X, Y)
    return X_train, X_test, y_train, y_test
    
def __splitData(_X, _Y):
    """
    Splits the X and Y datasets into training and testing datasets. The X
    datasets are represented as np arrays of lists, while Y datasets are lists.
    """
    X_train, X_test, y_train, y_test = train_test_split(_X, _Y, test_size = 0.2)
    print("Training Dimensions: ", X_train.shape, y_train.shape)
    print("Testing Dimensions: ", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main(sys.argv[1:])
