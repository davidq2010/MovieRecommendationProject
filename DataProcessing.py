
def parseUserDataFile(_userFile):
  """
  Read through _userFile and map the userID to a list containing Gender, Age,
  Occupation, ZipCode
  Occupation is represented by an occupation number from the README file

  Return this map of userID : List of features
  """

def parseMoveDataFile(_movieFile):
  """
  Read through _movieFile and map the movieID to a bitset array representing the
  genres that a movie falls under.

  Return this map of movieID : Genres bitset
  """

def parseRatingFile(_ratingFile):
  """
  Read through _ratingFile and map _userID and _movieID to a rating

  Return this map of tuple(userID, movieID) : rating
  """

def populateTrainingAndTestingMatrices(_userData, _movieData, _ratingData):
  """
  Randomly pick 80% of data for training matrices and remainder for testing
  matrices.

  Returns X_train, Y_train, X_test, Y_test matrices
  """
