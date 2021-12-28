import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam1.csv")

# Execute this if first 5 rows are to be checked
# print(df.head())
# Execute if the 'Category' column is to be checked for unique values
# print(df.groupby('Category').describe())

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

X_train, X_test, y_train, y_test = train_test_split(df.Message , df.spam)

clf.fit(X_train, y_train)

print(clf.score(X_test,y_test))

emails = [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
print(clf.predict(emails))

print(cross_val_score(clf.fit(X_train, y_train), X_train, y_train, cv=6))
