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
