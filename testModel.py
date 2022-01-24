#This file contains the code for training the model based on selected features and classfying some real mushrooms as poisonous or edible
#Refer MushroomClassification.ipynb for detailed working of the code
import pandas as pd 
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

new_features = ['odor', 'gill-color', 'gill-size', 'spore-print-color', 'gill-spacing']

df_new = df[['class'] + new_features]
X = df_new.drop(["class"], axis=1)
y = df_new["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test)*100, 2)))

#edible
agaricus_bisporus = {'Name': 'Agaricus Bisporous','gill-color': 'n', 'spore-print-color': 'n', 'gill-size': 'n', 'gill-spacing': 'w',
    'odor': 'n', 'ring-type': 'n'}
#poisonous
amanita_phalloides = {'Name': 'Amanita Phalloides','odor': 'f', 'gill-color': 'p',  'gill-size': 'b', 'spore-print-color': 'w',
    'ring-type': 'p', 'gill-spacing': 'w'}
#edible
volvariella_volvacea = {'Name': 'Volvariella Volvacea', 'gill-color': 'w', 'spore-print-color': 'w', 'gill-size': 'b', 'gill-spacing': 'c', 'odor': 'n'}
#poisonous
lepiota_cristata = {'Name': 'Lepiota Cristata', 'gill-color': 'w', 'spore-print-color':'w', 'gill-size':'b', 'gill-spacing': 'w', 'odor': 'f' }

mushrooms = [agaricus_bisporus, amanita_phalloides, volvariella_volvacea, lepiota_cristata]

for mushroom in mushrooms:
    for key in new_features:
        mushroom[key] = labelmap[key][mushroom[key]]

df_test = pd.DataFrame(mushrooms)

class_ = rf.predict(df_test[X_train.columns])

df_test['class'] = class_
print(f"\nLabel mapping of class: {labelmap['class']}")
print("e - edible \np - poisonous \n")
print(df_test[['Name', 'class']])