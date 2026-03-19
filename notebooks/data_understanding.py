import pandas as pd

# Load dataset
df = pd.read_csv("C:/Users/swaya/OneDrive/Desktop/gnn/fraud_detection_project/data/creditcard.csv")

# Show first 5 rows
print(df.head())

# Show dataset shape
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns)
# Count fraud vs normal
print(df['Class'].value_counts())
# Percentage
print(df['Class'].value_counts(normalize=True))
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()
print(df.info())
print(df.describe())