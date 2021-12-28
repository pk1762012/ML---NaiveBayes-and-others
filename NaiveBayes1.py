import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam1.csv")

# Execute this if first 5 rows are to be checked
# print(df.head())
# Execute if the 'Category' column is to be checked for unique values
# print(df.groupby('Category').describe())

df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

X_train, X_test, y_train, y_test = train_test_split(df.Message , df.spam)

v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

# Print the below if the array is to be checked
# print(X_train_count.toarray()[:2])

model = MultinomialNB()
model.fit(X_train_count, y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
model.predict(emails_count)

X_test_count = v.transform(X_test)
print(model.score(X_test_count, y_test))


#check prediction

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

print(cross_val_score(MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True), X_train_count, y_train, cv=6))
