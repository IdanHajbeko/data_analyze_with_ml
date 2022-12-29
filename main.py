#import liberys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

#with pandas we read the .csv file
Personality_data = pd.read_csv('psyc.csv')

#with pandas we clean and preper the data for use
x = Personality_data.drop(columns=['Personality'])
y = Personality_data[["Personality"]]

#we use the the model DecisionTreeClassifier from the sklearn.tree to make a machine learning model with the data from thhe .csv file
model = DecisionTreeClassifier()
model.fit(x, y)


#we save the model in a .joblib file
model = joblib.dump('Personality_teller.joblib')
#now we can just load the model from the .joblib file and dont need steps 1, 2, 3, and 4
#model = joblib.load('Personality_teller.joblib')

age = float(input("what you age: "))
openness = float(input("how open you are: "))
neuroticism = float(input("neuroticism: "))
conscientiousness = float(input("conscientiousness: "))
agreeableness = float(input("agreeableness: "))
extraversion = float(input("extraversion: "))

#we make a prediction with the machine learning model
prediction = model.predict([[age,openness,neuroticism,conscientiousness,agreeableness,extraversion]])
print(prediction)
