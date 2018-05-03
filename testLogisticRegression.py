#import sklearn.linear_model.LogisticRegression

def readDataFromFeatureFile(_file):
  f = open(_file, 'r')

  userList = list()

  lines = f.readlines()
  for line in lines:
    featureList = lineNew.split(' ')
    featureList.pop(len(featureList)-1)

    try:
    	featureList = [float(i) for i in featureList]
    	userList.append(featureList)
    except ValueError:
      #print("error on line ", line)
      print("")

  f.close()

  # Return our X feature list
  return userList


def readDataFromTargetFile(_file):
	f = open(_file, 'r')

	targetList = list()

	lines = f.readlines()

	for line in lines:
		targetList.append(float(line[0:1]))

	f.close()

	return targetList

#X = readDataFromFeatureFile('x_train.txt')

Y = readDataFromTargetFile('y_train.txt')

#print(X)
print(Y)
