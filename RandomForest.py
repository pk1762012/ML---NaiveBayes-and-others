# importing required libraries
# importing Scikit-learn library and datasets package
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Loading the iris plants dataset (classification)
iris = datasets.load_iris()

print(iris.target_names)
print(iris.feature_names)

# dividing the datasets into two parts i.e. training datasets and test datasets
X, y = datasets.load_iris(return_X_y=True)

# Spliting arrays or matrices into random train and test subsets
# i.e. 70 % training dataset and 30 % test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# creating a RF classifier
#clf = RandomForestClassifier(n_estimators=100)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
#clf.fit(X_train, y_train)

# performing predictions on the test dataset
#y_pred = clf.predict(X_test)

# using metrics module for accuracy calculation
#print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

clf=RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=50, cv= 5, n_jobs=-1)
CV_rfc.fit(X_train, y_train)

print(CV_rfc.best_params_)

clf2 = RandomForestClassifier(n_estimators=CV_rfc.best_params_.get("n_estimators"),
                             max_features=CV_rfc.best_params_.get("max_features"),
                             max_depth=CV_rfc.best_params_.get("max_depth"),
                             criterion=CV_rfc.best_params_.get("criterion"))
clf2.fit(X_train, y_train)
y_pred = clf2.predict(X_test)
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))


feature_imp = pd.Series(clf2.feature_importances_, index = iris.feature_names).sort_values(ascending = False)
print(feature_imp)
