#Import other necessary libraries like pandas and sklearn
import pandas as pd
from sklearn.metrics import accuracy_score

#Load Train and Test datasets
trainingSet = pd.read_csv('train.csv')
testingSet = pd.read_csv('test.csv')

#Replace data string to numeric 
trainingSet['Sex'].replace(['female','male'],[0,1],inplace=True)
testingSet['Sex'].replace(['female','male'],[0,1],inplace=True)

trainingSet['Embarked'].replace(['C','S','Q'],[0,1,2],inplace=True)
testingSet['Embarked'].replace(['C','S','Q'],[0,1,2],inplace=True)

trainingSet.describe()
#print(trainingSet.describe())

train = trainingSet.pop('Survived')
#print(train.head())

#Identify feature and response variable(s) and values must be numeric
numeric_variables=list(trainingSet.dtypes[trainingSet.dtypes != 'object'].index)
#print(trainingSet[numeric_variables].head())

#Cleaning for train data
trainingSet['Age'].fillna(trainingSet.Age.mean(), inplace=True)
trainingSet['Embarked'].fillna(trainingSet.Embarked.mean(), inplace=True)
#print(trainingSet.describe())

trainingSet.tail()
#print(trainingSet[numeric_variables].head())

#Import Library
from sklearn.ensemble import RandomForestClassifier

# Train the model using the training sets
model = RandomForestClassifier(n_estimators=100)
model.fit(trainingSet[numeric_variables],train)

# Train Accuracy
print('Train Accuracy : ', accuracy_score(train, model.predict(trainingSet[numeric_variables]))*100)

# cleaning for test data
testingSet['Age'].fillna(testingSet.Age.mean(), inplace=True)
trainingSet['Embarked'].fillna(trainingSet.Embarked.mean(), inplace=True)
testingSet = testingSet[numeric_variables].fillna(testingSet.mean()).copy()

#Predict data  
test_pred=model.predict(testingSet[numeric_variables])
#print(test_pred)

#Create submission file
submission=pd.DataFrame({"PassengerId": testingSet["PassengerId"], "Survived":test_pred})
submission.to_csv('submission_RF.csv',index=False)
print(submission.head())
