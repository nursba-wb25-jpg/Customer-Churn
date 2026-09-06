import pandas as pd

df = pd.read_csv('Telco_Customer_Churn.csv')
df.info()

display(df.head()) 
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv("Telco_Customer_Churn.csv")

# 2. Calculate counts
counts = df['Churn'].value_counts()

# 3. Create plot using Matplotlib
plt.figure(figsize=(7, 5))
colors = ['#2B5C8F', '#D95F02']  # Blue for 'No', Orange/Red for 'Yes'
bars = plt.bar(counts.index, counts.values, color=colors)

# 4. Annotate bars with exact counts and percentages
total = len(df)
for bar in bars:
    height = bar.get_height()
    pct = (height / total) * 100
    plt.annotate(f'{height:,}\n({pct:.1f}%)',
                 (bar.get_x() + bar.get_width() / 2., height + 100),
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold', color='#333333')

# 5. Format axes and title
plt.title('Customer Churn', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Churn Status', fontsize=12, labelpad=10)
plt.ylabel('Number of Customers', fontsize=12, labelpad=10)
plt.ylim(0, max(counts) * 1.18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Clustered Bar Chart for Categorical Features
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
cols = ['Contract', 'TechSupport', 'InternetService']

for i, col in enumerate(cols):
    sns.countplot(data=df, x=col, hue='Churn', ax=axes[i], palette='Set2')
    axes[i].set_title(f'Churn by {col}')

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# 2. Graph 1 (Top) - MUST include ax=axes[0], hue='Churn', and legend=False
sns.boxplot(
    data=df,
    x='Churn',
    y='MonthlyCharges',
    hue='Churn',
    palette=['red', '#0072B2'],
    legend=False,
     ax=axes[0],
)

sns.boxplot(
    data=df,
    x='Contract',
    y='Tenure',
    hue='Churn',
    palette=['#8DA0CB', '#FC8D62'],
    ax=axes[1],
)
axes[1].set_title(
    'Tenure by Contract Type and Churn Status', fontsize=12, fontweight='bold'
)
axes[1].set_xlabel('Contract Type')
axes[1].set_ylabel('Tenure (Months)')

plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('Telco_Customer_Churn.csv')
# 1. Load dataset
dataset["Contract_Code"] = dataset["Contract"].astype("category").cat.codes
dataset["TechSupport_Code"] = dataset["TechSupport"].astype("category").cat.codes
df_encoded = df.copy()

if 'tenure' in df_encoded.columns:
    df_encoded['Tenure'] = df_encoded['tenure']  # Handle case sensitivity

if 'Churn' in df_encoded.columns:
    df_encoded['Churn_Code'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

if 'Contract' in df_encoded.columns:
    df_encoded['Contract_Code'] = df_encoded['Contract'].astype('category').cat.codes

if 'TechSupport' in df_encoded.columns:
    df_encoded['TechSupport_Code'] = df_encoded['TechSupport'].astype('category').cat.codes

if 'InternetService' in df_encoded.columns:
    df_encoded['Internet_Code'] = df_encoded['InternetService'].astype('category').cat.codes

# 3. Compute full correlation matrix (added missing .corr() call)
corr_cols = ['Tenure', 'MonthlyCharges', 'Contract_Code', 'TechSupport_Code', 'Internet_Code', 'Churn_Code']
corr_matrix = df_encoded[corr_cols].corr()

# 4. Mask the upper triangle (including diagonal) for a clean lower-triangle plot
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 5. Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix, 
    mask=mask, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    vmin=-0.4, 
    vmax=0.8,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Between Each Variable of the Telco Dataset", fontsize=12, pad=12)
plt.tight_layout()
plt.show()

# 6. Generate and display descriptive statistics
stats_df = df_encoded[corr_cols].describe().round(2)
display(stats_df) 

import pandas as pd
import numpy as np

# 1. Load CSV while automatically converting all common missing placeholders to NaN
df = pd.read_csv('Telco_Customer_Churn.csv', na_values=['', ' ', 'N/A', 'NA', 'null', 'NULL', '?'])

# 2. Select target columns
cols = ['tenure', 'MonthlyCharges', 'TechSupport', 'Contract', 'InternetService', 'Churn']

# Fix casing if tenure is capitalized
cols = ['Tenure' if c == 'tenure' and 'Tenure' in df.columns else c for c in cols]

# 3. Complete missing summary report
missing_report = pd.DataFrame({
    'Missing_Count': df[cols].isnull().sum(),
    'Missing_Percentage': (df[cols].isnull().mean() * 100).round(2),
   'Data_Type': df[cols].dtypes
})

print(missing_report)

dataset.isnull().sum()

#heatmap for missing values
sns.heatmap(dataset.isnull(), cbar=False).set_title('Overview of Missing Values in this Telco Dataset')import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load Dataset
df = pd.read_csv('Telco_Customer_Churn.csv')

# Standardize 'tenure' column casing
if 'tenure' in df.columns:
  df.rename(columns={'tenure': 'Tenure'}, inplace=True)

# 2. Build df_encoded (converting string categories to numeric codes)
df_encoded = df.copy()
df_encoded['Contract_Code'] = (
    df_encoded['Contract'].astype('category').cat.codes
)
df_encoded['TechSupport_Code'] = (
    df_encoded['TechSupport'].astype('category').cat.codes
)
df_encoded['Internet_Code'] = (
    df_encoded['InternetService'].astype('category').cat.codes
   )
df_encoded['Churn_Code'] = (
    df_encoded['Churn'].map({'No': 0, 'Yes': 1}).fillna(0).astype(int)
)

# ------------------------------------------------------------------
# Step 2 - Dropping Unwanted Columns and Splitting Dataset (80:20)
# ------------------------------------------------------------------
dataset_train, dataset_test = train_test_split(
    df_encoded.drop(
        columns=['Contract', 'TechSupport', 'InternetService', 'Churn']
    ),
    test_size=0.2,
    random_state=42,
)


# ------------------------------------------------------------------
# Step 3 - Separating train set and test set into X and y respectively
# ------------------------------------------------------------------
def dataset_seperator(dataset):
  X = dataset.drop('Churn_Code', axis=1)
  y = dataset['Churn_Code']
  return X, y
   dataset_X_train, dataset_y_train = dataset_seperator(dataset_train)
dataset_X_test, dataset_y_test = dataset_seperator(dataset_test)


# Function to show shape and feature columns of train and test sets
def print_feature(X_train, X_test):
  print('X train shape:', X_train.shape, r'\X test shape:', X_test.shape)
  print(X_train.shape[1], 'features:', list(X_train.columns))


print('dataset:')
print_feature(dataset_X_train, dataset_X_test)

dataset_y_train = pd.DataFrame(dataset_y_train)
dataset_y_test = pd.DataFrame(dataset_y_test)

dataset_y_train.columns = ['Churn']
dataset_y_test.columns = ['Churn']

mapping = {'Yes': 1, 'No': 0}
dataset_y_train['Churn'] = dataset_y_train['Churn'].replace(mapping)
dataset_y_test['Churn'] = dataset_y_test['Churn'].replace(mapping)

dataset_y_train.head(5)
def percentile_displayer(datasets):
  percentile = np.zeros(shape=(0, 2))
  for feature in datasets.columns:
    ten = round(np.percentile(datasets[feature], 10), 2)
    nine = round(np.percentile(datasets[feature], 90), 2)


    print(f'{feature:<18} - Percentile of 10th: {ten:>8.2f}  | - Percentile of 90th: {nine:>8.2f}')
    percentile = np.append(percentile, [[ten, nine]], axis=0)

  print()
  return percentile


# Print percentile values for training and testing datasets
print('Percentile in Telco training set:')
dataset_percentile_train = percentile_displayer(dataset_X_train)

print('Percentile in Telco testing set:')
dataset_percentile_test = percentile_displayer(dataset_X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Fit the model
logreg = LogisticRegression()
logreg.fit(dataset_X_train, dataset_y_train.values.ravel())

# predict the test and train result
y_pred = logreg.predict(dataset_X_test)
y_pred_train = logreg.predict(dataset_X_train)
print(
    "Testing set score Logistic Regression (Telco) :",
    accuracy_score(dataset_y_test, y_pred),
)

# Check training accuracy
print("\n")
print(
    "Training set score Logistic Regression (Telco):", accuracy_score(dataset_y_train, y_pred_train),)


import numpy as np
from sklearn.metrics import mean_squared_error

mse_list=[]
rmse_list=[]

# Calculate Mean Squared Error and Root Mean Squared Error
mseLrTelco = mean_squared_error(dataset_y_test, y_pred)
rmseLrTelco = np.sqrt(mseLrTelco)

print("Telco:")
print("Mean Squared Error for Logistic Regression Telco:", mseLrTelco)
print("\n")
print("Root Mean Squared Error for Logistic Regression Telco:", rmseLrTelco) 


# push both mse and rmse into respective array
mse_list.append(mseLrTelco)
rmse_list.append(rmseLrTelco)
if 'accuracy_list' not in globals():
    accuracy_list = []

accLrTelco = accuracy_score(dataset_y_test, y_pred)

# Push the accuracy score into accuracy list array
accuracy_list.append(accLrTelco)

# Print the Accuracy of Logistic Regression
print('Accuracy of Logistic Regression is : ', '{:.2f}%'.format(100 * accLrTelco))
print('\n')

# Classification Report
print('Classification Report for Logistic Regression:')
logreg_classification_report = classification_report(dataset_y_test, y_pred)
print(logreg_classification_report)

# Display Graph or Visualize the model using ConfusionMatrixDisplay
cm7 = confusion_matrix(dataset_y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm7, display_labels=['No - 0', 'Yes - 1']
)

disp.plot(cmap=plt.cm.PuRd)
plt.title('Logistic Regression')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
neighbor = np.arange(1, 15)
train_dataset_Accuracy = np.empty(len(neighbor))
test_dataset_Accuracy = np.empty(len(neighbor))

for i, k in enumerate(neighbor):
    # setup knn classifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # fit the model
    knn.fit(dataset_X_train, dataset_y_train.values.ravel())
    
    # compute accuracy on train data
    train_dataset_Accuracy[i] = knn.score(dataset_X_train, dataset_y_train.values.ravel())
    
    # compute accuracy on test data
    test_dataset_Accuracy[i] = knn.score(dataset_X_test, dataset_y_test.values.ravel())

# generate plot
plt.title('KNN Varying number of K neighbors for Telco')
plt.plot(neighbor, test_dataset_Accuracy, label='Testing Accuracy')
plt.plot(neighbor, train_dataset_Accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(dataset_X_train, dataset_y_train)

# Predict the test and train result
pred_test_y = knn.predict(dataset_X_test)
pred_train_y = knn.predict(dataset_X_train)

# Check testing accuracy
print(
    "Testing set score KNN (Telco) :", accuracy_score(dataset_y_test, pred_test_y)
)

# Check training accuracy
print("\n")
print(
    "Training set score KNN (Telco):",
    accuracy_score(dataset_y_train, pred_train_y),
)
import numpy as np
from sklearn.metrics import mean_squared_error

pred_test_y = knn.predict(dataset_X_test)

# Calculate Mean Squared Error and Root Mean Squared Error

print("Telco:")
mseKnnTelco = mean_squared_error(dataset_y_test, pred_test_y)
rmseKnnTelco = np.sqrt(mseKnnTelco)

print("Mean Squared Error for KNN Telco:", mseKnnTelco)
print("\n")
print("Root Mean Squared Error for KNN Telco:", rmseKnnTelco)

# Push both mse and rmse into respective array
if "mse_list" not in globals():
  mse_list = []
if "rmse_list" not in globals():
  rmse_list = []

mse_list.append(mseKnnTelco)
rmse_list.append(rmseKnnTelco)

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Classification Report (Making the confusion matrix and calculating accuracy score)
accKnnTelco = accuracy_score(dataset_y_test, pred_test_y)

# Push the accuracy score of KNN into accuracy list array
if "accuracy_list" not in globals():
  accuracy_list = []
accuracy_list.append(accKnnTelco)

# Print the accuracy of KNN
print("Accuracy of KNN is : ", "{:.2f}%".format(100 * accKnnTelco))
print("\n")

# Print Classification Report
print("Classification Report for KNN:")
knn_classification_report = classification_report(dataset_y_test, pred_test_y)
print(knn_classification_report)

# Display Graph or Visualize the model
# Confusion matrix
cm = confusion_matrix(dataset_y_test, pred_test_y)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No - 0", "Yes - 1"]
)
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN Model Confusion Matrix")
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Find the best c score for SVC
cScores = []
c_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for c in c_values:
  classifier = SVC(C=c, random_state=0, kernel='rbf')
  classifier.fit(dataset_X_train, dataset_y_train.values.ravel())
 y_pred2 = classifier.predict(dataset_X_test)
  cScores.append(accuracy_score(dataset_y_test.values.ravel(), y_pred2))

plt.figure()
plt.plot(c_values, cScores)
plt.xlabel('C Value')
plt.ylabel('Accuracy')
plt.title('SVC Varying C Value for Telco')
plt.show()

svmClassifier = SVC(C=0.6, random_state=0, kernel='rbf')
svmClassifier.fit(dataset_X_train, dataset_y_train.values.ravel())

# predict the test and train result
pred_test_y_svm = svmClassifier.predict(dataset_X_test)
pred_train_y_svm = svmClassifier.predict(dataset_X_train)

# Check testing accuracy
print(
    "Testing set score SVM (Telco) :",
    accuracy_score(dataset_y_test, pred_test_y_svm),
)

# Check training accuracy
print("\n")
print(
    "Training set score SVM (Telco):",
    accuracy_score(dataset_y_train, pred_train_y_svm),
)
import numpy as np
from sklearn.metrics import mean_squared_error

# Calculate Mean Squared Error and Root Mean Squared Error SVM
print("Telco:")
mseSvmTelco = mean_squared_error(dataset_y_test, pred_test_y_svm)
rmseSvmTelco = np.sqrt(mseSvmTelco)

print("Mean Squared Error for SVM Telco:", mseSvmTelco)
print("\n")
print("Root Mean Squared Error for SVM Telco:", rmseSvmTelco)

# push both mse and rmse into respective array
if "mse_list" not in globals():
  mse_list = []
if "rmse_list" not in globals():
  rmse_list = []

mse_list.append(mseSvmTelco)
rmse_list.append(rmseSvmTelco) 

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Classification Report (Making the confusion matrix and calculating accuracy score)
accSvmTelco = accuracy_score(dataset_y_test, pred_test_y_svm)

# push the accuracy score of SVM into accuracy list array
if 'accuracy_list' not in globals():
  accuracy_list = []
accuracy_list.append(accSvmTelco)

# print the Accuracy of SVM
print('Accuracy of SVM is : ', '{:.2f}%'.format(100 * accSvmTelco))

# Print Classification Report
print('\n')
print('Classification Report for SVM:') 
print(classification_report(dataset_y_test, pred_test_y_svm))

# Display Graph or Visualize the model
# Confusion Matrix
cm = confusion_matrix(dataset_y_test, pred_test_y_svm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=['No - 0', 'Yes - 1']
)
disp.plot(cmap=plt.cm.Oranges)
plt.title('SVM Model - Confusion Matrix')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Ensure accuracy scores are defined from test predictions
accLrTelco = accuracy_score(dataset_y_test, y_pred)
accKnnTelco = accuracy_score(dataset_y_test, pred_test_y)
accSvmTelco = accuracy_score(dataset_y_test, pred_test_y_svm)

# Define model names and accuracy values
models = ['LogisticRegression', 'KNearestNeighbours', 'SupportVector']
accuracies = [accLrTelco, accKnnTelco, accSvmTelco]

# Plot bar chart for Accuracy comparison
plt.figure(figsize=(7, 5))
bars = plt.bar(
    models,
    [a * 100 for a in accuracies],
    color=['purple', 'crimson', 'sandybrown'],
)
# Annotate values on top of bars
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.5,
      f'{yval:.2f}%',
      ha='center',
      va='bottom',
  )

plt.title('Accuracy of Different Classifier Models')
plt.xlabel('Classifier Models')
plt.ylabel('% of Accuracy')
plt.ylim(0, 100)
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Compute F1-Scores for positive class (Churn = 1)
f1LrTelco = f1_score(dataset_y_test, y_pred)
f1KnnTelco = f1_score(dataset_y_test, pred_test_y)
f1SvmTelco = f1_score(dataset_y_test, pred_test_y_svm)

# Define model names and F1-score values
models = ['LogisticRegression', 'KNearestNeighbours', 'SupportVector']
f1_scores = [f1LrTelco, f1KnnTelco, f1SvmTelco]
# Plot bar chart for F1-Score comparison
plt.figure(figsize=(7, 5))
bars = plt.bar(
    models,
    [f * 100 for f in f1_scores],
    color=['purple', 'crimson', 'sandybrown'],
)

# Annotate values on top of bars
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.5,
      f'{yval:.2f}%',
      ha='center',
      va='bottom',
  )

plt.title('F1-Score of Different Classifier Models')
plt.xlabel('Classifier Models')
plt.ylabel('% of F1-Score')
plt.ylim(0, 100)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# Recalculate RMSE directly from model predictions
rmseLrTelco = np.sqrt(mean_squared_error(dataset_y_test, y_pred))
rmseKnnTelco = np.sqrt(mean_squared_error(dataset_y_test, pred_test_y))
rmseSvmTelco = np.sqrt(mean_squared_error(dataset_y_test, pred_test_y_svm))

# Define model names and exact RMSE values
models = ['LogisticRegression', 'KNearestNeighbours', 'SupportVector']
rmse_values = [rmseLrTelco, rmseKnnTelco, rmseSvmTelco]

# Plot bar chart for Root Mean Squared Error comparison
plt.figure(figsize=(7, 5))
bars = plt.bar(models, rmse_values, color=['purple', 'crimson', 'sandybrown'])

# Annotate values on top of bars
for bar in bars:
  yval = bar.get_height()
  plt.text(
      bar.get_x() + bar.get_width() / 2.0,
      yval + 0.01,
      f'{yval:.4f}',
      ha='center',
      va='bottom',
  )

plt.title('Root Mean Squared Error of Different Classifier Models')
plt.xlabel('Classifier Models')
plt.ylabel('Root Mean Squared Error')
plt.ylim(0, max(rmse_values) * 1.25)
plt.show()

accuracies_dict = {'Logistic Regression': accLrTelco, 'KNearestNeighbours': accKnnTelco, 'SupportVector': accSvmTelco}
mse_dict = {'Logistic Regression': mseLrTelco, 'KNearestNeighbours': mseKnnTelco, 'SupportVector': mseSvmTelco}
rmse_dict = {'Logistic Regression': rmseLrTelco, 'KNearestNeighbours': rmseKnnTelco, 'SupportVector': rmseSvmTelco}

# Determining the best and worst models
great_acc = max(accuracies_dict, key=accuracies_dict.get)
low_acc = min(accuracies_dict, key=accuracies_dict.get)

great_mse = min(mse_dict, key=mse_dict.get)
low_mse = max(mse_dict, key=mse_dict.get)

great_rmse = min(rmse_dict, key=rmse_dict.get)
low_rmse = max(rmse_dict, key=rmse_dict.get)

# Print final evaluation summary
print("Determining the best model in terms of accuracy, Mean Squared Error and Root Mean Squared Error\n")
print(f"Best Model based on Accuracy: {great_acc}")
print(f"Worst Model based on Accuracy: {low_acc}\n")

print(f"Best Model based on Mean Squared Error: {great_mse}")
print(f"Worst Model based on Mean Squared Error: {low_mse}\n")

print(f"Best Model based on Root Mean Squared Error: {great_rmse}")
print(f"Worst Model based on Root Mean Squared Error: {low_rmse}") 


import pickle
import streamlit as st
from sklearn.linear_model import LogisticRegression 
logleg = LogisticRegression()
logreg.fit(dataset_X_train,dataset_y_train)
# Define file path for pickle object
Pkl_filename = "Pickle_LogReg_Model.pkl"

# Save the model to file in the current working directory
with open(Pkl_filename, 'wb') as file:
   Pickled_LogReg_Model = pickle.load(file)

# Reload the model from file
with open(Pkl_filename, 'rb') as file:
    Pickled_LogReg_Model = pickle.load(file)

# Calculate test accuracy score using reloaded model
score = Pickled_LogReg_Model.score(dataset_X_test, dataset_y_test.values.ravel())
print("Test score: {:.2f}%".format(100 * score))

# Predict target labels using the reloaded model
ypredicting = Pickled_LogReg_Model.predict(dataset_X_test)
print(ypredicting)

# Deploying the top-performing Logistic Regression model
def result_predictor(inputs):
    # Process missing values by defaulting to 0
    cleaned_inputs = [0 if (val == "" or val is None) else float(val) for val in inputs]
    
    # Predict outcome using reloaded pickle model
    prediction = Pickled_LogReg_Model.predict([cleaned_inputs])
    
    if prediction[0] == 1:
        return "Customer will Churn (Yes)"
    else:
        return "Customer will Not Churn (No)"
def main():
    print("--- Telco Churn Prediction System ---")
    p1 = input("Enter Tenure (months): ")
    p2 = input("Enter Monthly Charges ($): ")
    p3 = input("Enter Contract Type (0=M2M, 1=1Yr, 2=2Yr): ")
    p4 = input("Enter Tech Support (0=No, 1=No Internet, 2=Yes): ")
    p5 = input("Enter Internet Service (0=DSL, 1=Fiber, 2=No): ")
    
    machineResult = result_predictor(p1, p2, p3, p4, p5)
    print("Final Predicted Machine Result:", machineResult)

# Run interactive system
main() 
