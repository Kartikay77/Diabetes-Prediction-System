# -*- coding: utf-8 -*-
"""Diabetes prediction system.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1T5WOW3y4WCqMAPIBCHkHimomfE6VsKbu

# Diabetes disease analysis and prediction system
"""

from google.colab import files
uploaded = files.upload()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
df=pd.read_csv('diabetes_prediction_dataset.csv')
df.head()

"""# Data Cleansing

#### Listing Null Values form data
"""

print(df.isnull().sum())

df=df.dropna()

df.info()

df.describe()

"""#### Diabetes Stats"""

d=df['diabetes'].value_counts()
print(d)
No_Disease = len(df[df['diabetes'] ==0])
Diseased = len(df[df['diabetes'] ==1])

print('Percentage of No_Disease: {:.2f} %' .format(No_Disease/len(df['diabetes'])*100))
print('Percentage of Diseased: {:.2f} %' .format(Diseased/len(df['diabetes'])*100))

"""#### age Stats"""

min_age = df['age'].min()
max_age = df['age'].max()
mean_age = round(df['age'].mean(),1)

print('Min age: %s' %min_age)
print('Max age: %s' %max_age)
print('Mean age: %s' %mean_age)

"""#### Sex Stats"""

female = len(df[df['gender'] =='Female'])
male = len(df[df['gender'] =='Male'])

print('Percentage of female: {:.2f} %' .format(female/len(df['gender'])*100))
print('Percentage of male: {:.2f} %' .format(male/len(df['gender'])*100))

"""# Data Visualization"""

df.hist(bins=20,figsize=(15,15),grid=True,ec='black',color='violet')
plt.show()

plt.figure(figsize=(15, 12))
sns.pairplot(
    df[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']],
    hue='diabetes',
    palette='icefire'
)
plt.show()

"""##### diabetes Diseases Ratio in Dataset
###### Blue Graph indicate no diabetes desease and Orange Graph show diabetes desease
"""

def plotDiabetes():
    sns.countplot(x='diabetes', data=df, ax=ax)
    for i, p in enumerate(ax.patches):
        count = df['diabetes'].value_counts().values[i]
        x = p.get_x() + p.get_width() / 2.
        y = p.get_height() + 3
        label = '{:1.2f}'.format(count / float(df.shape[0]))
        ax.text(x, y, label, ha='center')

fig_diabetes, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 2))
plotDiabetes()

from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

df_encoded.corr()

"""##### Select Age as most dependent data on label
###### Disease Probability Bar Plot
"""

def plotAge():
    # KDE Plot with FacetGrid
    facet_grid = sns.FacetGrid(df, hue='diabetes')
    facet_grid.map(sns.kdeplot, "age", fill=True)  # use fill instead of shade
    facet_grid.add_legend()  # ensures legend is available

    # Rename legend labels safely
    legend_labels = ['No Diabetes', 'Diabetes']
    for t, l in zip(facet_grid._legend.texts, legend_labels):
        t.set_text(l)

    facet_grid.set_axis_labels('age', 'Density')

    # Barplot of average diabetes probability vs age
    avg = df[["age", "diabetes"]].groupby(['age'], as_index=False).mean()
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.barplot(x='age', y='diabetes', data=avg, ax=ax)
    ax.set(xlabel='age', ylabel='Diabetes Probability')

plotAge()

plt.figure(figsize=(12, 10))

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Plot heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap="magma", fmt='.2f')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

"""##### Checking For Categorical Data"""

x = df['gender']
x.value_counts()

x = df['smoking_history']
x.value_counts()

category = [
    ('gender', ['Female', 'Male', 'Other']),
    ('smoking_history', ['No Info', 'never', 'former', 'current', 'not current', 'ever'])
]

continuous = [
    ('age', 'Age in years'),
    ('bmi', 'Body Mass Index'),
    ('HbA1c_level', 'HbA1c level'),
    ('blood_glucose_level', 'Blood Glucose Level')
]

def plotCategorial(attribute, labels, ax_index):
    sns.countplot(x=attribute, data=df, ax=axes[ax_index][0])
    sns.countplot(x='diabetes', hue=attribute, data=df, ax=axes[ax_index][1])
    avg = df[[attribute, 'diabetes']].groupby([attribute], as_index=False).mean()
    sns.barplot(x=attribute, y='diabetes', hue=attribute, data=avg, ax=axes[ax_index][2])

    # Update legend labels only if legend exists
    if axes[ax_index][1].get_legend() is not None:
        for t, l in zip(axes[ax_index][1].get_legend().texts, labels):
            t.set_text(l)

    if axes[ax_index][2].get_legend() is not None:
        for t, l in zip(axes[ax_index][2].get_legend().texts, labels):
            t.set_text(l)

def plotGrid(isCategorial):
    if isCategorial:
        [plotCategorial(x[0], x[1], i) for i, x in enumerate(category)]
    else:
        [plotContinuous(x[0], x[1], i) for i, x in enumerate(continuous)]

import matplotlib.pyplot as plt

# For categorical features (2 rows, 3 cols to leave space)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
plotGrid(isCategorial=True)

# For continuous features (4 rows, 2 cols)
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16))
plotGrid(isCategorial=False)

"""##### Creating Dummy"""

# Create dummy variables for 'gender'
gender_dummy = pd.get_dummies(df['gender'])
gender_dummy.rename(columns={'Female': 'Gender_Female', 'Male': 'Gender_Male', 'Other': 'Gender_Other'}, inplace=True)

# Create dummy variables for 'smoking_history'
smoke_dummy = pd.get_dummies(df['smoking_history'])
smoke_dummy.rename(columns={
    'never': 'Smoke_Never',
    'former': 'Smoke_Former',
    'current': 'Smoke_Current',
    'No Info': 'Smoke_NoInfo',
    'not current': 'Smoke_NotCurrent',
    'ever': 'Smoke_Ever'
}, inplace=True)

# Concatenate the dummy variables into the dataframe
df = pd.concat([df, gender_dummy, smoke_dummy], axis=1)

# Drop original columns after encoding
df.drop(['gender', 'smoking_history'], axis=1, inplace=True)

df.info()

df.head()

df_X= df.loc[:, df.columns != 'diabetes']
df_y= df.loc[:, df.columns == 'diabetes']

"""# Model Training"""

import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import warnings

# Assuming df_X contains features and df_y contains the target variable
selected_features = []
lr = LogisticRegression()
n_features_to_select = 20  # Define the number of features to select

rfe = RFE(estimator=lr, n_features_to_select=n_features_to_select)

warnings.simplefilter('ignore')
rfe.fit(df_X.values, df_y.values)
print(rfe.support_)
print(rfe.ranking_)

for i, feature in enumerate(df_X.columns.values):
    if rfe.support_[i]:
        selected_features.append(feature)

df_selected_X = df_X[selected_features]
df_selected_y = df_y

lm = sm.Logit(df_selected_y, df_selected_X)
result = lm.fit()

print(result.summary2())
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(df_selected_X,df_selected_y, test_size = 0.25, random_state =0)
columns = X_train.columns

"""##### Calculating Accuracy Function and confusion Matrix of the Models"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
def cal_accuracy(y_test, y_predict):

    print("\nConfusion Matrix: \n",
    confusion_matrix(y_test, y_predict))

    print (f"\nAccuracy : {accuracy_score(y_test,y_predict)*100:0.3f}")

"""# Logistic Regression"""

lr=LogisticRegression()
lr.fit(X_train,y_train)
print(f"Accuracy of Test Dataset: {lr.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {lr.score(X_train,y_train):0.3f}")
warnings.simplefilter('ignore')

"""##### Vale Prediction for Test dataset for Logistic Regression"""

y_predict=lr.predict(X_test)
print("Predicted values:")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# Support Vector Machine"""

from sklearn import svm
svm_linear = svm.SVC(kernel='linear')
svm_linear.fit(X_train,y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {svm_linear.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {svm_linear.score(X_train,y_train):0.3f}")

"""##### Value Prediction for Test dataset for SVM"""

y_predict=svm_linear.predict(X_test)
print("Predicted values:")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# Decision Tree"""

from sklearn.tree import DecisionTreeClassifier
gini = DecisionTreeClassifier(criterion = "gini", random_state =100,max_depth=3, min_samples_leaf=5)
gini.fit(X_train, y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {gini.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {gini.score(X_train,y_train):0.3f}")

"""##### Value Prediction for Test dataset for Decision Tree"""

y_predict=gini.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Desicion Tree Diagram"""

from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(decision_tree=gini, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = X_test.columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('heart-disease-analysis-prediction.png')
Image(graph.create_png())

"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)
forest.fit(X_train,y_train)

warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {forest.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {forest.score(X_train,y_train):0.3f}")

"""##### Over Fitting Issue
##### Vale Prediction for Test dataset for Rondom Forest
"""

y_predict=forest.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# Naive Bayes"""

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {nb.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {nb.score(X_train,y_train):0.3f}")

"""##### Vale Prediction for Test dataset for Naive Bayes"""

y_predict = nb.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# K Nearest Neighbor(KNN)"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
warnings.simplefilter('ignore')

classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# XGBoost"""

import xgboost as xgb
XGB = xgb.XGBClassifier()
XGB.fit(X_train,y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {XGB.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {XGB.score(X_train,y_train):0.3f}")

y_predict=XGB.predict(X_test)
print('Predicted values:\n')
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# AdaBoost"""

from sklearn.ensemble import AdaBoostClassifier
ABC=AdaBoostClassifier()
ABC.fit(X_train,y_train)
warnings.simplefilter('ignore')
print(f"Accuracy of Test Dataset: {ABC.score(X_test,y_test):0.3f}")
print(f"Accuracy of Train Dataset: {ABC.score(X_train,y_train):0.3f}")

y_predict=ABC.predict(X_test)
print('Predicted values:\n')
print(y_predict)
cal_accuracy(y_predict,y_test)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# Artificial Neural Network (ANN)"""

from keras.models import Sequential
from keras.layers import Dense

ANN = Sequential(name='DCNN')
ANN.add(Dense(11, activation='relu', input_dim=X_train.shape[1]))  # dynamically match input size
ANN.add(Dense(1, activation='sigmoid'))

ANN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ANN.fit(X_train,y_train,epochs=250)

y_predict=ANN.predict(X_test)
print('Predicted values:\n')
rounded = [round(x[0]) for x in y_predict]
y_predict=rounded
print(y_predict)
cal_accuracy(y_predict,y_test)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""# Cross Validation For Models"""

from sklearn import model_selection
kfold=model_selection.KFold(n_splits=10)
models=[('Logistic Regression', lr), ('Support Vector Machine', svm_linear), ('Decision Tree', gini),
        ('Random Forest', forest), ('K Nearest Neighbor', classifier),('XGBoost',XGB),('AdaBoostClassifier', ABC)]
warnings.simplefilter('ignore')

for model in models:
    results=model_selection.cross_val_score(model[1],X_train,y_train,cv=kfold,scoring='accuracy')
    print(f"Cross validated Accuracy of  {model[0]}: {results.mean():.3f}")

models=pd.DataFrame({'Model':['Logistics Regression','Support Vector Machine','Decision Tree','Random Forest','Naive Bayes','K Nearest Neighbor','eXtreme Gradient Boosting','AdaBoost'],
                     'Traning Accuracy':[(lr.score(X_train,y_train)),svm_linear.score(X_train,y_train),gini.score(X_train,y_train),forest.score(X_train,y_train),nb.score(X_train,y_train),classifier.score(X_train,y_train),XGB.score(X_train,y_train),ABC.score(X_train,y_train)],
                     'Test Accuracy':[(lr.score(X_test,y_test)),svm_linear.score(X_test,y_test),gini.score(X_test,y_test),forest.score(X_test,y_test),nb.score(X_test,y_test),classifier.score(X_test,y_test),XGB.score(X_test,y_test),ABC.score(X_test,y_test)]})
models.sort_values(by='Test Accuracy', ascending=False)

x=[lr.score(X_test,y_test),svm_linear.score(X_test,y_test),gini.score(X_test,y_test),forest.score(X_test,y_test),nb.score(X_test,y_test),classifier.score(X_test,y_test),XGB.score(X_test,y_test),ABC.score(X_test,y_test)]
y=['Logistics Regression','Support Vector Machine','Decision Tree','Random Forest','Naive Bayes','K Nearest Neighbor','eXtreme Gradient Boosting','AdaBoost']
plt.scatter(x,y)

import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {'LR':81.5789,'SVM':69.7368,'DT':69.737, 'RF':84.2105,'NB':85.5263, 'KNN':82.8947,'XGBoost':84.2105,'AdaBoost':78.9474,'ANN':85.526}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10,5))

# creating the bar plot
plt.bar(courses, values, color ='maroon',width = 0.4)
for i in range(len(courses)):
    plt.text(i,values[i],values[i])

plt.xlabel("ML Algorithms")
plt.ylabel("Accuracy Scores")
plt.title("Accuracy of diffrent ML Algorithms")
plt.show()

"""## Gridsearch on Random Forest to increase the accuracy"""

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [2,4]
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf
             }

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, verbose=2)

grid.fit(X_train, y_train)

grid.best_params_

model = RandomForestClassifier(max_depth=4,max_features='sqrt',min_samples_leaf=2,min_samples_split=2,n_estimators=25)

model.fit(X_train,y_train)

y_predict = model.predict(X_test)
print(y_predict)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for random forest  """

model.score(X_test,y_test)

"""## Gridsearch on Logistic Regression to increase the accuracy"""

from sklearn.model_selection import GridSearchCV

grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
lr_cv=GridSearchCV(lr,grid,cv=10)
lr_cv.fit(X_train, y_train)

lr_cv.best_params_

y_predict=lr_cv.predict(X_test)
print(y_predict)
print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for Logistic Regression"""

lr_cv.score(X_test, y_test)

"""## Gridsearch on Support Vector Machine to increase the accuracy"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(X_train, y_train)

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)

y_predict=grid.predict(X_test)
print("Predicted values:")
print(y_predict)
print("\nConfusion Matrix: \n",confusion_matrix(y_test, y_predict))
print("\nClassification Report: \n",classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for Support Vector Machine"""

grid.score(X_test,y_test)

"""## Randomized Search on eXtreme Gradient Boosting to increase the accuracy"""

xgb_classifier = xgb.XGBClassifier()

gbm_param_grid = {
    'n_estimators': range(1,20),
    'max_depth': range(1, 10),
    'learning_rate': [.1,.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1],
    'booster':["gbtree"],
     'min_child_weight': [0.001,0.003,0.01],
}

from sklearn.model_selection import RandomizedSearchCV
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid,
                                    estimator = xgb_classifier, scoring = "accuracy",
                                    verbose = 0, n_iter = 100, cv = 4)

xgb_random.fit(X_train, y_train)

xgb_bp = xgb_random.best_params_

xgb_model=xgb.XGBClassifier(n_estimators=xgb_bp["n_estimators"],
                            min_child_weight=xgb_bp["min_child_weight"],
                            max_depth=xgb_bp["max_depth"],
                            learning_rate=xgb_bp["learning_rate"],
                            colsample_bytree=xgb_bp["colsample_bytree"],
                            booster=xgb_bp["booster"])

xgb_model.fit(X_train, y_train)

y_predict=xgb_model.predict(X_test)
print("Predicted values:")
print(y_predict)
print("\nConfusion Matrix: \n",confusion_matrix(y_test, y_predict))
print("\nClassification Report: \n",classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for  XGBoost (eXtreme Gradient Boosting)"""

xgb_model.score(X_test, y_test)

"""## Gridsearch on AdaBoost to increase accuracy"""

shallow_tree = DecisionTreeClassifier(max_depth=1, random_state = 100)

from sklearn import metrics
estimators = list(range(20,25))

abc_scores = []
for n_est in estimators:
    ABC = AdaBoostClassifier(
    estimator=shallow_tree,
    n_estimators = n_est)

    ABC.fit(X_train, y_train)
    y_pred = ABC.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    abc_scores.append(score)

# plot test scores and n_estimators
# plot
plt.plot(estimators, abc_scores)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.70, 1])
plt.show()

ABC = AdaBoostClassifier(
    estimator=shallow_tree,
    n_estimators = 21)

ABC.fit(X_train, y_train)
y_predict = ABC.predict(X_test)

print("Predicted values:")
print(y_predict)
print("\nConfusion Matrix: \n",confusion_matrix(y_test, y_predict))
print("\nClassification Report: \n",classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for  AdaBoost"""

print(accuracy_score(y_test,y_predict))

"""## Gridsearch on Neural Network to increase accuracy"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
def create_binary_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=20, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, kernel_initializer='normal',  kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    adam = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

binary_model = create_binary_model()

print(binary_model.summary())

history=binary_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=10)

y_predict=binary_model.predict(X_test)
print('Predicted values:\n')
rounded = [round(x[0]) for x in y_predict]
y_predict = rounded
print(y_predict)
cal_accuracy(y_predict,y_test)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

"""### Accuacy increased for Neural Network"""

print(accuracy_score(y_test,y_predict))

"""## Gridsearch on k-Nearest Neighbours to increase accuracy"""

KNN = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
params_KNN = dict(n_neighbors = range(1,10))
grid_search_KNN = GridSearchCV(KNN, param_grid = params_KNN, cv =4, scoring='recall')
grid_search_KNN.fit(X_train,y_train)

KNN_best_k = grid_search_KNN.best_params_['n_neighbors']
print("For a k-Nearest Neighbors model, the optimal value of k is "+str(KNN_best_k))
KNN_df = pd.DataFrame(grid_search_KNN.cv_results_)
fig_KNN = plt.figure(figsize=(12,9))
plt.plot(KNN_df['param_n_neighbors'],KNN_df['mean_test_score'],'b-o')
plt.xlim(0,10)
plt.ylim(0.5,1.0)
plt.xlabel('k')
plt.ylabel('Mean recall over 4 cross-validation sets')

classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
classifier = classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)
print("Predicted values:\n")
print(y_predict)
cal_accuracy(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(confusion_matrix(y_test,y_predict),annot=True)

print(accuracy_score(y_test,y_predict))

models=pd.DataFrame({'Model':['Random Forest','Logistics Regression','eXtreme Gradient Boosting','AdaBoost','SVM'],
                     'Traning Accuracy':[(model.score(X_train,y_train)),lr_cv.score(X_train,y_train),xgb_model.score(X_train,y_train),ABC.score(X_train,y_train),grid.score(X_train,y_train)],
                     'Test Accuracy':[(model.score(X_test,y_test)),lr_cv.score(X_test,y_test),xgb_model.score(X_test,y_test),ABC.score(X_test,y_test),grid.score(X_test,y_test)]})
models.sort_values(by='Test Accuracy', ascending=False)

import numpy as np
import matplotlib.pyplot as plt


# creating the dataset
data = {'RF':85.526315,'LR':88.157894,'SVM':89.473684, 'XGBoost':86.842105,'AdaBoost':81.578947,'ANN':88.157894,'KNN':86.842105}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10,5))

# creating the bar plot
plt.bar(courses, values, color ='maroon',width = 0.4)
for i in range(len(courses)):
    plt.text(i,values[i],values[i])

plt.xlabel("ML Algorithms with Hyperparameter tuning")
plt.ylabel("Increased Accuracy Scores")
plt.title("Increased Accuracy of diffrent ML Algorithms")
plt.show()

"""# Feature Importance

### Logistic regression
"""

# Fit the instance of LogisticRegression
clf = LogisticRegression(C=0.38566204211634725,
                        solver="liblinear")
clf.fit(X_train, y_train);

# Checking coefficients
clf.coef_

# Match coef's of features to columns

feature_dictionary_lr = dict(zip(df.columns, list(clf.coef_[0])))
feature_dictionary_lr

# Visualize the feature importance
feature_df = pd.DataFrame(feature_dictionary_lr, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);

"""### Decision Tree"""

gini.feature_importances_

feature_dictionary_dt = dict(zip(df.columns, list(gini.feature_importances_)))
feature_dictionary_dt

# Visualize the feature importance
feature_df = pd.DataFrame(feature_dictionary_dt, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);

"""### Random Forest"""

forest.feature_importances_

feature_dictionary_rf = dict(zip(df.columns, list(forest.feature_importances_)))
feature_dictionary_rf

# Visualize the feature importance
feature_df = pd.DataFrame(feature_dictionary_rf, index=[0])
feature_df.T.plot.bar(title="Feature Importance", legend=False);

"""### XGBoost (eXtreme Gradient Boosting)"""

XGB.feature_importances_

feature_dictionary_XGB = dict(zip(df.columns, list(XGB.feature_importances_)))
feature_dictionary_XGB

# Visualize the feature importance
feature_XGB = pd.DataFrame(feature_dictionary_XGB, index=[0])
feature_XGB.T.plot.bar(title="Feature Importance", legend=False);

"""### AdaBoost"""

ABC.feature_importances_

feature_dictionary_ABC = dict(zip(df.columns, list(ABC.feature_importances_)))
feature_dictionary_ABC

# Visualize the feature importance
feature_XGB = pd.DataFrame(feature_dictionary_XGB, index=[0])
feature_XGB.T.plot.bar(title="Feature Importance", legend=False);

"""# For manulally inputting data and finding whether Heart Disease or not"""

print("Enter Patient's Name:")
name = input()

print("Enter Patient's Age:")
age = int(input())

print("Enter Patient's Gender (1=Male, 0=Female):")
gender_male = int(input())
gender_female = 1 - gender_male
gender_other = 0  # you can extend this if needed

print("Enter Patient's Body Mass Index (BMI):")
bmi = float(input())

print("Enter Patient's HbA1c level:")
hba1c = float(input())

print("Enter Patient's Blood Glucose Level:")
glucose = float(input())

print("Does the patient have Hypertension? (1=Yes, 0=No):")
hypertension = int(input())

print("Does the patient have any Heart Disease? (1=Yes, 0=No):")
heart_disease = int(input())

print("Enter Patient's Smoking History:")
print("Options: 1=never, 2=former, 3=current, 4=No Info, 5=not current, 6=ever")
smoke_input = int(input())

# One-hot encode smoking history
smoke_never = int(smoke_input == 1)
smoke_former = int(smoke_input == 2)
smoke_current = int(smoke_input == 3)
smoke_noinfo = int(smoke_input == 4)
smoke_notcurrent = int(smoke_input == 5)
smoke_ever = int(smoke_input == 6)

# Input vector (order must match training data)
arr = [[
    age,
    bmi,
    hba1c,
    glucose,
    hypertension,
    heart_disease,
    gender_female,
    gender_male,
    gender_other,
    smoke_never,
    smoke_former,
    smoke_current,
    smoke_noinfo,
    smoke_notcurrent,
    smoke_ever
]]

# Prediction
x = grid.predict(arr)[0]  # 0 = No diabetes, 1 = Diabetes

# Probability (assumes model supports predict_proba)
perc = str(int(model.predict_proba(arr)[0, 1] * 100))

# Output message
print('Hello ' + name + '!')
if x == 0:
    print('You are not diabetic.')
    print('You have only ' + perc + '% chance of having diabetes, which is normal for a healthy person.')
else:
    print('You may have diabetes.')
    print('You have a high ' + perc + '% chance of having diabetes. Please consult a doctor as soon as possible.')

print("Enter Patient's Name:")
name = input()

print("Enter Patient's Age:")
age = int(input())

print("Enter Patient's Gender (1=Male, 0=Female):")
gender_male = int(input())
gender_female = 1 - gender_male
gender_other = 0  # you can extend this if needed

print("Enter Patient's Body Mass Index (BMI):")
bmi = float(input())

print("Enter Patient's HbA1c level:")
hba1c = float(input())

print("Enter Patient's Blood Glucose Level:")
glucose = float(input())

print("Does the patient have Hypertension? (1=Yes, 0=No):")
hypertension = int(input())

print("Does the patient have any Heart Disease? (1=Yes, 0=No):")
heart_disease = int(input())

print("Enter Patient's Smoking History:")
print("Options: 1=never, 2=former, 3=current, 4=No Info, 5=not current, 6=ever")
smoke_input = int(input())

# One-hot encode smoking history
smoke_never = int(smoke_input == 1)
smoke_former = int(smoke_input == 2)
smoke_current = int(smoke_input == 3)
smoke_noinfo = int(smoke_input == 4)
smoke_notcurrent = int(smoke_input == 5)
smoke_ever = int(smoke_input == 6)

# Input vector (order must match training data)
arr = [[
    age,
    bmi,
    hba1c,
    glucose,
    hypertension,
    heart_disease,
    gender_female,
    gender_male,
    gender_other,
    smoke_never,
    smoke_former,
    smoke_current,
    smoke_noinfo,
    smoke_notcurrent,
    smoke_ever
]]

# Prediction
x = grid.predict(arr)[0]  # 0 = No diabetes, 1 = Diabetes

# Probability (assumes model supports predict_proba)
perc = str(int(model.predict_proba(arr)[0, 1] * 100))

# Output message
print('Hello ' + name + '!')
if x == 0:
    print('You are not diabetic.')
    print('You have only ' + perc + '% chance of having diabetes, which is normal for a healthy person.')
else:
    print('You may have diabetes.')
    print('You have a high ' + perc + '% chance of having diabetes. Please consult a doctor as soon as possible.')