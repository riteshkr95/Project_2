import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("/users/riteshkumar/Desktop/Data_Science/Datasets/anemia_dataset.csv")

#print(df)

df=df[["%Red Pixel" ,"%Green pixel" ,"%Blue pixel" ,"Hb" ,"Anaemic"]]

#print(df)

# EDA
print(
df.isna().sum())

print(df.duplicated().sum())

# one Hot Encoding

df["Anaemic"]=pd.get_dummies(df["Anaemic"] ,drop_first=True ,dtype=int)

# variable separations

X=df.drop(columns="Anaemic")
y=df["Anaemic"]

# train ,test and split
X_train ,X_test ,y_train, y_test=train_test_split(X ,y ,test_size=0.3 ,random_state=43)

# model
model=LogisticRegression()
model.fit(X_train ,y_train)

y_pred=model.predict(X_test)

#print(y_pred)

from sklearn.metrics import classification_report

report=classification_report(y_pred ,y_test)
print(report)

# import model as pkl file

joblib.dump(model,"log_reg.pkl")