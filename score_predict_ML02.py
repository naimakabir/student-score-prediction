import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

print(df.columns)

# Select only the necessary columns
data = df[['Hours_Studied', 'Attendance', 'Exam_Score']]

# Check for missing values
print("\nMissing values:\n", data.isnull().sum())

print("\nStatistics insight:\n", data.describe())

# Select features and target
X = data[['Hours_Studied', 'Attendance']] 
y = data['Exam_Score']

# Split the dataset -- 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Function for predicting student's exam score
def predict_score(hours_studied, attendance):
    if not (0 <= attendance <= 100):
        return "Attendance must be between 0 and 100"
    
    # Create a DataFrame with the input values & corresponding columns
    # As "predict()"method takes a single argument/a 2D array/a list of lists/a DataFrame
    # and error would raise if input values directly use
    input_data = pd.DataFrame([[hours_studied, attendance]], 
                             columns=['Hours_Studied', 'Attendance'])
  
    prediction = model.predict(input_data)
    return f"Predicted exam score: {prediction[0]:.1f}"

# Example 
print("\nExample Predictions:")
print(predict_score(20, 85))
print(predict_score(30, 95))
print(predict_score(10, 70))  

# Plotting actual vs predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Scores")
plt.show()