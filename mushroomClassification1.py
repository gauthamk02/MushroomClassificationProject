#from tkinter import Y
import numpy as np 
#import seaborn as sns 
#import os 
#import graphviz
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
#from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve

import pandas as pd 
import matplotlib.pyplot as plt 
import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("mushrooms.csv")

features = ['class', 'odor', 'gill-color', 'gill-size', 'spore-print-color', 'gill-spacing']
#features = ['class', 'gill-attachment', 'cap-shape']
print(df['gill-spacing'].unique())
df = df[features]
features.remove('class')

df = df.astype('category')

labelmap = {}
labelencoder=LabelEncoder()

for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
    labelmap[column] = dict(zip(labelencoder.classes_,range(len(labelencoder.classes_))))

X = df.drop(["class"], axis=1)
y = df["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)
#print(X.head())
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
['class', 'odor', 'gill-color', 'gill-size', 'spore-print-color', 'gill-spacing']
#edible 
agaricus_bisporus = {'Name': 'Agaricus Bisporous','gill-color': 'n', 'spore-print-color': 'n', 'gill-size': 'n', 'gill-spacing': 'w',
    'odor': 'n', 'ring-type': 'n'} #'population': 'a', 

#poisonous
amanita_phalloides = {'Name': 'Amanita Phalloides','odor': 'f', 'gill-color': 'p',  'gill-size': 'b', 'spore-print-color': 'w','ring-type': 'p', 'gill-spacing': 'w'}#,'ring-type': 'p'}#, 'population': 'y'}

#poisonous
lepiota_cristata = {'Name': 'Lepiota Cristata', 'odor': 'f', 'gill-color': 'w', 'gill-size': }
#poisonous
#lepiota_magnispora = {'Name': 'Lepiota Magnispora', 'odor': 'n', 'gill-color': 'w', 'gill-size': 'n', 'spore-print-color': 'w', 'gill-spacing': 'd'}

mushrooms = [agaricus_bisporus, amanita_phalloides, lepiota_magnispora]#, amanita_phalloides]

pprint.pprint(labelmap)

for mushroom in mushrooms:
    for key in features:
        mushroom[key] = labelmap[key][mushroom[key]]

df2 = pd.DataFrame(mushrooms)

pred = rf.predict(df2[X_test.columns])
print(pred)

df2['class'] = pred
print(labelmap['class'])
print(df2[['Name','class']])



