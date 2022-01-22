from tkinter import Y
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 
#import graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

print("Started")

df = pd.read_csv("mushrooms.csv")

#df = df[['class', 'gill-color', 'spore-print-color', 'population', 'gill-size',
#        'stalk-root', 'habitat']]

features = ['class', 'odor', 'gill-color', 'gill-size', 'spore-print-color', 'gill-spacing']

df = df[features]
features.remove('class')
#print(df.head())

df = df.astype('category')

labelmap = {}
labelencoder=LabelEncoder()

for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
    labelmap[column] = dict(zip(labelencoder.classes_,range(len(labelencoder.classes_))))

X = df.drop(["class"], axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
print(X.head())
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

"""
features_list = X.columns.values
feature_importance = rf.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(8,7))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color ="red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
#plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')
plt.show()"""


n = 1
#pred = rf.predict(X_test.iloc[n:n+1])[0]

#for key, val in labelmap['class'].items():
#    if val == pred:
#        print(key)

#edible 
agaricus_bisporus = {'name': 'Agaricus Bisporous','gill-color': 'n', 'spore-print-color': 'n', 'gill-size': 'n', 'gill-spacing': 'w',
    'odor': 'n'} #'population': 'a', 

#poisonous
amanita_phalloides = {'name': 'Amanita Phalloides','odor': 'f', 'gill-color': 'p',  'gill-size': 'b', 'spore-print-color': 'w', 'gill-spacing': 'w'}#,'ring-type': 'p'}#, 'population': 'y'}


mushrooms = [agaricus_bisporus, amanita_phalloides]

#print(y_test.iloc[n:n+1])

print(labelmap)

#print(X_test.iloc[n:n+1])


for mushroom in mushrooms:
    for key in features:
        mushroom[key] = labelmap[key][mushroom[key]]
print(mushrooms)
df2 = pd.DataFrame(mushrooms)
print(df2)
#df2.set_index('name', inplace= True)
print(df2[X_test.columns])
pred = rf.predict(df2[X_test.columns]) 
print(pred)
df2['class'] = pred
print(df2[['name','class']])



