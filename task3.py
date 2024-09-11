import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Customer Churn dataset
url = "task3.csv"
df = pd.read_csv(url)

# Feature engineering (replace with your chosen features)
# Example: Convert categorical data to numerical
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
for col in categorical_cols:
    df[col] = pd.Categorical(df[col]).codes  # Example using category codes

# Feature selection (replace with your chosen features)
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]  # Example features

# Target variable
y = df["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# (Optional) Visualize the decision tree (requires additional libraries)
from sklearn.tree import export_graphviz
export_graphviz(model, out_file="tree.dot", feature_names=X.columns)