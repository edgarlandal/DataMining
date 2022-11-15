from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid

def decision_Tree(X, Y, X_test):
    clf = tree.DecisionTreeClassifier()
    return clf.fit(X, Y).predict(X_test)     

def vector_machine(X, Y, X_test):
    clf = svm.SVC()
    return clf.fit(X, Y).predict(X_test)
       
def sgd(X, Y, X_test):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
    return clf.fit(X, Y).predict(X_test)     
     
def NNmodels(X, Y, X_test):
    clf = MLPClassifier(solver='adam', alpha=1e-5,  hidden_layer_sizes=(10, 6), random_state=2, max_iter=100)
    return clf.fit(X, Y).predict(X_test)     

def neighbors(X, Y, X_test):
    clf = NearestCentroid() 
    return clf.fit(X, Y).predict(X_test)     
