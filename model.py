import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load data
file_path = 'data.csv'
data = pd.read_csv(file_path)
# data = pd.read_csv("https://raw.githubusercontent.com/firmansyahken/datasets/main/tb.csv")

# Drop coloms
data = data.drop(columns=['KECAMATAN'])
data = data.drop(columns=['HASIL TCM'])

# Data preprocessing
scaler = MinMaxScaler()
data['UMUR'] = scaler.fit_transform(data[['UMUR']])
# data['UMUR'] = pd.cut(data['UMUR'], bins=[0, 41, 83], labels=[1, 0])
data['JENIS KELAMIN'] = data['JENIS KELAMIN'].replace({'L': 1, 'P': 0})
data['FOTO TORAKS'] = data['FOTO TORAKS'].replace({'Positif':1,'Negatif':0,'Tidak dilakukan': pd.NA})
data['STATUS HIV'] = data['STATUS HIV'].replace({'Positif':1,'Negatif':0,'Tidak diketahui': pd.NA}) 
data['RIWAYAT DIABETES'] = data['RIWAYAT DIABETES'].replace({'Ya':1,'Tidak':0,'Tidak diketahui': pd.NA}) 
data['LOKASI ANATOMI (target/output)'] = data['LOKASI ANATOMI (target/output)'].replace({'Paru':1, 'Ekstra paru': 0})

data['FOTO TORAKS'] = data['FOTO TORAKS'].fillna(data['FOTO TORAKS'].mode()[0])
data['STATUS HIV'] = data['STATUS HIV'].fillna(data['STATUS HIV'].mode()[0])
data['RIWAYAT DIABETES'] = data['RIWAYAT DIABETES'].fillna(data['RIWAYAT DIABETES'].mode()[0])

data = data.dropna(subset=['LOKASI ANATOMI (target/output)'])
X = data[['UMUR', 'JENIS KELAMIN', 'FOTO TORAKS', 'STATUS HIV', 'RIWAYAT DIABETES']]
y = data['LOKASI ANATOMI (target/output)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

# Save the accuracy to a file
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

# Export a decision tree from the forest
estimator = model.estimators_[0]
export_graphviz(estimator, out_file='tree.dot', 
                feature_names=['UMUR', 'JENIS KELAMIN', 'FOTO TORAKS', 'STATUS HIV', 'RIWAYAT DIABETES'],
                class_names=['Paru', 'Ekstra paru'],
                rounded=True, proportion=False, 
                precision=2, filled=True)

# Convert to png
call(['dot', '-Tpng', 'tree.dot', '-o', 'static/tree.png', '-Gdpi=600'])
