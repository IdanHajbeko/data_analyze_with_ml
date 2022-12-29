import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data[['genre']]

model = DecisionTreeClassifier()
model.fit(x, y)

#model = joblib.load('music-recommender.joblib')
age = float(input("what is your age: "))
gender = float(input("are you a man or a woman 1 for man 0 for woman: "))
prediction = model.predict([[age, gender]])
print(prediction)