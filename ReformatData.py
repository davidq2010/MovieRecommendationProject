def parseUserDataFile(_userFile):
  """
  Read through _userFile and map the userID to a list containing Gender, Age,
  Occupation, ZipCode
  Occupation is represented by an occupation number from the README file

  Return this map of userID : List of features
  """
  users = {}
  f = open(_userFile, 'r')
  lines = f.readlines()
  for line in lines:
    features = []
    tokens = line.split("::")

    # Encode gender variable, 1 is female, 0 is male
    if(tokens[1] == 'F'):
      features.append('Female')
    else:
      features.append('Male')

    # Age
    features.append(tokens[2])
    # Occupation
    features.append(tokens[3])
    # Zip code region
    features.append(tokens[4][0])

    users[tokens[0]] = features

  f.close()
  return(users)

def parseMovieDataFile(_movieFile):
  """
  Read through _movieFile and map the movieID to a bitset array representing the
  genres that a movie falls under.

  Return this map of movieID : Genres bitset
  """
  genres = ['Action','Adventure','Animation','Children\'s','Comedy','Crime','Documentary',
  'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical','Mystery','Romance','Sci-Fi',
  'Thriller','War','Western']
  movies = {}

  f = open(_movieFile, 'r', encoding = "ISO-8859-1")
  lines = f.readlines()

  for line in lines:
    features = []
    tokens = line.split("::")

    thisGenres = tokens[2].split("|")
    i = 0
    while i < len(genres):
      flag = False
      for genre in thisGenres:
        if genre == genres[i]:
          features.append('1')
          flag = True
      if not flag:
        features.append('0')
      i = i + 1

    movies[tokens[0]] = features

  f.close()
  return (movies)

# Retriever user and movie data
users = parseUserDataFile('users.dat')
movies = parseMovieDataFile('movies.dat')
ratingFile = 'ratings.dat'

with open('ratings.dat','r') as ratings, open('x_data.csv', 'w') as x, open('y_data.csv', 'w') as y:
  x.write('Gender,Age,Occupation,Region,Action,Adventure,Animation,Children\'s,Comedy,Crime,Documentary,' +
  'Drama,Fantasy,Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western' + '\n')
  y.write('Rating' + '\n')
  lines = ratings.readlines()
  for line in lines:
    tokens = line.split("::")
    userFeatures = users[tokens[0]]
    movieFeatures = movies[tokens[1]]
    for feature in userFeatures:
      x.write(feature + ',')

    i = 0
    while i < len(movieFeatures) - 1:
      x.write(movieFeatures[i] + ',')
      i = i + 1
    x.write(movieFeatures[i] + '\n')

    y.write(tokens[2] + '\n')

