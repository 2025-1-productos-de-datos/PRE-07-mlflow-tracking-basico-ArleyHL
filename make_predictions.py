import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y=df["quality"]
x=df.drop(columns=["quality"])

logged_model="runs:/ce231329d4bd4ff79373f32caa5535db/model"
loaded_model=mlflow.pyfunc.load_model(logged_model)
y=loaded_model.predict(x)
print(y)
