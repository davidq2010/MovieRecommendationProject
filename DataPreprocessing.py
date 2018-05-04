from sklearn.preprocessing import OneHotEncoder
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
    X = pd.read_csv(featureFile)
    Y = pd.read_csv(targetFile)
    

if __name__ == "__main__":
    main(sys.argv[1:])
