#from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys, getopt
from sklearn.model_selection import train_test_split

def __readDataFromFeatureFile(_file):
  """
  Private function to read data from feature file and store them into a list
  of lists.
  """
  print("Reading Feature file...")
  f = open(_file, 'r')

  userList = list()

  lines = f.readlines()
  for line in lines:
    featureList = line.split(' ')
    featureList.pop(len(featureList)-1)

    try:
    	featureList = [float(i) for i in featureList]
    	userList.append(featureList)
    except ValueError:
      #print("error on line ", line)
      print("")

  f.close()
  df = pd.DataFrame(data = userList)

  # Return our X feature list
  return df


def __readDataFromTargetFile(_file):
  """
  Private function to read data from target file and store it into a list.
  """
  print("Reading Target file...")
  f = open(_file, 'r')

  targetList = list()
  lines = f.readlines()

  for line in lines:
    targetList.append(float(line[0:1]))

  f.close()
  df = pd.DataFrame(data = targetList)

  return df

def getXYdataFrames(_featureFile, _targetFile):
    X = __readDataFromFeatureFile(_featureFile)
    Y = __readDataFromTargetFile(_targetFile)
    return X, Y

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
    X,Y = getXYdataFrames(featureFile, targetFile)
    print(X)
    print(Y)


if __name__ == "__main__":
    main(sys.argv[1:])
