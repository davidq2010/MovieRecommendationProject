#import sklearn.linear_model.LogisticRegression
import numpy as np
import pandas as pd

def readDataFromFeatureFile(_file):
  '''
  Read data from feature file and store them into a list of list

  '''
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


def readDataFromTargetFile(_file):
  f = open(_file, 'r')

  targetList = list()
  lines = f.readlines()

  for line in lines:
    targetList.append(float(line[0:1]))
  
  f.close()
  df = pd.DataFrame(data = targetList)
	
  return df

X = readDataFromFeatureFile('x_train.txt')
Y = readDataFromTargetFile('y_train.txt')

print(X)
print(Y)
