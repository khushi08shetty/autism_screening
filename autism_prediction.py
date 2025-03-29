import pandas as pd
df = pd.read_csv('autism_screening.csv')

df = df.drop(['contry_of_res','used_app_before','relation','age_desc','ethnicity'],axis=1)

import numpy as np
df.replace('?', np.nan, inplace=True)
df.fillna(df['age'].median(), inplace=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Map categorical values to 0 and 1
df["gender"] = df["gender"].map({"f": 0, "m": 1})
df["jundice"] = df["jundice"].map({"no": 0, "yes": 1})
df["austim"] = df["austim"].map({"no": 0, "yes": 1})

df.drop(['result','gender','jundice','austim'], axis=1, inplace=True)
X = df.drop(columns=["Class/ASD"])
y = df["Class/ASD"].map({"YES": 1, "NO": 0})  # Convert to binary

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')