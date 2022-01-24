#This file contains the code for training the model based on all features in the dataset, finding its accuracy and feature importance.
#Refer MushroomClassification.ipynb for detailed working of the code
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("mushrooms.csv")

df = df.astype('category')

labelmap = {}
labelencoder=LabelEncoder()

for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
    labelmap[column] = dict(zip(labelencoder.classes_,range(len(labelencoder.classes_))))

X = df.drop(["class"], axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

features_list = X.columns.values
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()

#Uncomment to save the Image
#plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')

plt.show()