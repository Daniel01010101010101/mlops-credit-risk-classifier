
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["path"])
df.dropna(inplace=True)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Married"] = le.fit_transform(df["Married"])
df["Education"] = le.fit_transform(df["Education"])
df["Self_Employed"] = le.fit_transform(df["Self_Employed"])
df["Property_Area"] = le.fit_transform(df["Property_Area"])
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["model"]["random_state"])

model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["model"]["random_state"]
)

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_param("n_estimators", config["model"]["n_estimators"])
    mlflow.log_param("max_depth", config["model"]["max_depth"])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
