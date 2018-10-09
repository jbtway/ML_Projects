from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class Knn_Classifier():

    def __init__(self, k=5):
        self.k = k
        self.num_inputs = 0
        self.x_train = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_test = np.array([])
        self.y_predict = np.array([])

    def fit(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_inputs = np.shape(x_test)[0]


    def predict(self):
        for i in range(self.num_inputs):
            distances = np.sum((self.x_train - self.x_test[i,]) ** 2, axis = 1)
            dist_sorted = np.sort(distances)

            indices = np.array([])
            for j in range(self.k):
                temp = np.where(distances == dist_sorted)
                indexes = np.append(indices, temp)

            nearest_k = np.array([])
            for j in range(len(indices)):
                nearest_val = self.y_train[indices[j]]
                nearest_k = np.append(nearest_k, nearest_val)

            unique, counts = np.unique(nearest_k, return_counts=True)
            weights = np.array((unique, counts)).T
            weights = np.asarray([weights[:,1].argsort()])
            answer = weights[-1]

            self.y_predict = np.append(self.y_predict, answer)

        return self.y_predict

def main():

    #Load data
    x = datasets.load_iris().data
    y = datasets.load_iris().target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=7)

    #Scale attributes
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = Knn_Classifier(k=5)
    classifier.fit(x_train, y_train, x_test, y_test)
    y_predict = classifier.predict()

    #accuracy = accuracy_score(y_test, y_predict).round(5)

    print(y_test.size, y_predict.size)

if __name__ == "__main__":
    main()
