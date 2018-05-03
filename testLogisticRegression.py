#import sklearn.linear_model.LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


