# Train classifier

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dic=pickle.load(open('./data.pickle','rb')) # loads data


data=np.asarray(data_dic['data'])
label=np.asarray(data_dic['label'])

X_train,X_test,y_train,y_test=train_test_split(data,label,test_size=0.2,shuffle=True,stratify=label) # split data for training and testing

model=RandomForestClassifier()   # training fast
model.fit(X_train,y_train) # train model
y_predict=model.predict(X_test)

score=accuracy_score(y_predict,y_test)

print('{}% of samples were classified correctly !'.format(score * 100)) # check accuracy
# stores model
f=open('model.p','wb')
pickle.dump({'model':model},f)
f.close()
