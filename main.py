# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import pandas as pd
df=pd.read_csv("C:/Users/Home/Downloads/mushrooms.csv")
df.head
df.columns
df.info()
df.isnull().sum()
df['class'].value_counts()
df[df['class']=='e'].describe().T
df[df['class']=='p'].describe().T
X=df.drop('class',axis=1)
y=df['class']
from sklearn.preprocessing import LabelEncoder
Encoder_x=LabelEncoder()
for col in X.columns:
    X[col]=Encoder_x.fit_transform(X[col])
Encoder_y=LabelEncoder()
y=Encoder_y.fit_transform(y)
print(y)
y.head()
X.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=54)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_predict=rfc.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,rfc_predict))
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
print(accuracy_score(y_test,rfc_predict))
print(precision_score(y_test,rfc_predict))