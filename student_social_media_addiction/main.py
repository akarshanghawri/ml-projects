import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv('student_data.csv')

# Drop ID (not useful for prediction)
df = df.drop('Student_ID', axis=1)

# Let's say Addicted_Score >= 6 means addicted
df['Addicted'] = df['Addicted_Score'].apply(lambda x: 1 if x >= 6 else 0)
df = df.drop('Addicted_Score', axis=1)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])

# Split data
X = df.drop('Addicted', axis=1)
y = df['Addicted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model,'model.pkl')
