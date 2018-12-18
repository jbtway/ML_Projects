#####################################
# IMPORT NEEDED LIBRARIES
#####################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

#CREATE LABEL ENCODER
label_encoder = LabelEncoder()

##################################################
# GET GLASS DATA AND NORMALIZE IT
##################################################

def glass_prep(data_path):
    
    #CREATE HEADER LIST
    header = ["ID", "Refractive Index", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass Type"]

    #READ IN DATA
    data = pd.read_csv(data_path,
                        header=None, names=header)

    #GET VALUES
    glass = data.values

    #GET ATTRIBUTES
    X = glass[:, 0:-1]
    
    #GET CLASS
    Y = glass[:, -1]

    #NORMALIZE DATA
    X_norm = normalize(X, norm="l1")
    
    return X_norm, Y

##################################################
# GET CENSUS DATA AND NORMALIZE IT
##################################################

def census_prep(data_path):
    
    #CREATE HEADERS FOR ATTRIBUTES
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
              "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
              "salary"]
    
    #GET DATA AND FORMAT
    census = pd.read_csv(data_path,
                         header=None, names=header)
    census = census.replace("?", np.nan)
    census = census.dropna()
    census = census.values
    
    #GET ATTRIBUTES
    X = census[:, 0:-1]
    
    #GET CLASS
    Y = census[:, -1]
     
    #ENCODE APPROPRIATE COLUMNS
    X[:, 1] = label_encoder.fit_transform(X[:, 1])
    X[:, 3] = label_encoder.fit_transform(X[:, 3])
    X[:, 5] = label_encoder.fit_transform(X[:, 5])
    X[:, 6] = label_encoder.fit_transform(X[:, 6])
    X[:, 7] = label_encoder.fit_transform(X[:, 7])
    X[:, 8] = label_encoder.fit_transform(X[:, 8])
    X[:, 9] = label_encoder.fit_transform(X[:, 9])
    X[:, 13] = label_encoder.fit_transform(X[:, 13])
    
    #NORMALIZE DATA
    X = normalize(X)
    
    return X, Y

##################################################
# GET AUTISM DATA AND NORMALIZE IT
##################################################

def screening_prep(data_path):
    
    #CREATE HEADER LIST
    header = ["Question 1",        "Question 2",        "Question 3",
              "Question 4",        "Question 5",        "Question 6",
              "Question 7",        "Question 8",        "Question 9",
              "Question 10",       "Age",               "Gender",
              "Ethnicity",         "Born w/ Jaundice",  "Fam Mem w/ PDD",  
              "Country of Res",    "Used Before",       "Screening Score",
              "Age Description",   "Relation",          "Class/ASD"]

    #GET DATA AND VALUES
    autism_screen = pd.read_csv(data_path, header=None, names=header)
    autism_screen = autism_screen.replace("?", np.nan)
    autism_screen = autism_screen.dropna()
    autism_screen = autism_screen.values
    
    #GET ATTRIBUTES
    X = autism_screen[:, 0:-1]
    
    #GET CLASS
    Y = autism_screen[:, -1]
    
    #ENCODE APPROPRIATE COLUMNS
    X[:, 10] = label_encoder.fit_transform(X[:, 10])
    X[:, 11] = label_encoder.fit_transform(X[:, 11])
    X[:, 12] = label_encoder.fit_transform(X[:, 12])
    X[:, 13] = label_encoder.fit_transform(X[:, 13])
    X[:, 14] = label_encoder.fit_transform(X[:, 14])
    X[:, 15] = label_encoder.fit_transform(X[:, 15])
    X[:, 16] = label_encoder.fit_transform(X[:, 16])
    X[:, 18] = label_encoder.fit_transform(X[:, 18])
    X[:, 19] = label_encoder.fit_transform(X[:, 19])
    
    #NORMALIZE DATA
    X = normalize(X)
    
    return X, Y

##################################################
# BUILD TRAINING AND TEST DATASETS THEN BUIL NN
# AND MEASURE ACCURACY
##################################################

def train_NN(X, Y):
  
    # SEPARATE TRAINING AND TEST SETS
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 7)
    
    #CREATE NN OBJECT
    classifier = MLPClassifier(max_iter=300, random_state=15, learning_rate_init=0.001,
                               activation="logistic")
    
    #FIT MODEL WITH TRAINING DATA
    model = classifier.fit(x_train, y_train)

    #MAKE PREDICTIONS FOR TRAINING SET
    predictions = classifier.predict(x_test)

    #GET ACCURACY FROM TEST SET
    accuracy = classifier.score(x_test, y_test)
    
    return accuracy


def main():
    
    #CREATE MODEL AND GENERATE ACCURACIES FOR GLASS DATA
    glass_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    glass_x, glass_y = glass_prep(glass_path)
    g_accuracy = train_NN(glass_x, glass_y)
    
    #CREATE MODEL AND GENERATE ACCURACIES FOR CENSUS DATA
    census_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    census_x, census_y = census_prep(census_path)
    c_accuracy = train_NN(census_x, census_y)
    
    #CREATE MODEL AND GENERATE ACCURACIES FOR AUTISM DATA
    autism_path = "Autism-Adult-Data.csv"
    autism_x, autism_y = screening_prep(autism_path)
    a_accuracy = train_NN(autism_x, autism_y)
    
    print("Accuracy for Glass:\n{0:.2f}%".format(g_accuracy*100))
    print("Accuracy for Census:\n{0:.2f}%".format(c_accuracy*100))
    print("Accuracy for Autism:\n{0:.2f}%".format(a_accuracy*100))
    
    
if __name__ == "__main__":
    main()
    
#NOTE: I DID WORK WITH KIM ON THIS ASSIGNMENT