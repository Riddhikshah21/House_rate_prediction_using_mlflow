import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import mlflow
import pickle
from mlflow.pyfunc import PythonModel

df = pd.read_csv("AmesHousing.csv")

#selected features: Lot area, Gr Liv Area, Garage Area, Blg type, Saleprice
#Blg type is a categorical value so need to convert it to numerical value using One hot encoding.
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_array = encoder.fit_transform(df[['Bldg Type']])
encoded_df = pd.DataFrame(
    encoded_array, 
    columns=encoder.get_feature_names_out(['Bldg Type'])
)
data = pd.concat([df, encoded_df], axis=1)
# data

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Model exercise:")

#load dataset 
data = pd.read_csv("AmesHousing.csv")
feature_columns = ['Lot Area','Gr Liv Area', 'Garage Area', 'Bldg Type']
selected_feature = data.loc[:, feature_columns + ['SalePrice']]

categorical_feature = ['Bldg Type']

#custom the model using pyfunc flavor
class WrapperLRModel(PythonModel):

    def __init__(self, sklearn_model_features, categorical_feature, model_artifact_name):
        """
        categorical_features: Mapping from categorical feature names to all possible 
        values, e.g:
        {
        "Bldg Type": ["1Fam", ...]
        }
        """
        self.feature_names = sklearn_model_features
        self.categorical_features = categorical_feature
        self.model_artifact_name = model_artifact_name

    def _encode(self, row, colname):
        value = row[colname]
        row[value] = 1
        return row

    def predict(self, context, model_input, params = None):
        model_features = model_input
        for col, unique_values in self.categorical_features.itmes():
            for uv in unique_values:
                model_features[uv] = 0
            model_features = model_features.apply(lambda row: self._encode(row, col), axis=1)
        model_features = model_features.loc[:, self.feature_names]
        return self.lr_model.predict(model_features.to_numpy())
    
    def load_context(self, context):
        with open(context.artifacts[self.model_artifact_name], 'rb') as m:
            self.lr_model = pickle.load(m)


def prepare_data(data):
    df = data
    categorical_feature_values = {}
    for col in list(data.columns):
        if col in categorical_feature:
            categorical_feature_values[col] = list(data[col].unique())
            dummies = pd.get_dummies(df[col])
            df = pd.concat([df.drop([col], axis=1), dummies], axis=1)
    df = df.fillna(0)
    return df, categorical_feature_values


def train_and_evaluate(data, categorical_feature):
    features = data.drop(['SalePrice'], axis=1).to_numpy()
    target = data.loc[:, 'SalePrice'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    #plot 
    plot = data.plot.scatter(x=0, y="SalePrice")
    fig = plot.get_figure()
    fig.savefig("tmp/plot.png")

    #save the data  
    data.to_csv("tmp/dataset.csv", index=False)

    #train the model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    #save the model 
    serialized_model = pickle.dumps(model)
    with open("tmp/model.pkl", "wb") as f:
        f.write(serialized_model)

    model_artifact_name = "original_sklearn_model"
    model_artifacts = {model_artifact_name: "tmp/model.pkl"}
    

    # #using sklearn flavor to log the model 
    # mlflow.sklearn.log_model(model, "model")

    #using custom model with pyfunc flavor
    mlflow.pyfunc.log_model(
        "custom_model",
        python_model=WrapperLRModel(sklearn_model_features=list(features.columns)),
        model_artifact_name = model_artifacts,
        artifacts=model_artifacts
    )

    #tsaved single artifact file but cannot visualize its contents in UI as its a binary file
    mlflow.log_artifact("tmp/model.pkl", artifact_path="models")

    #evaluate the model
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE:", err)

with mlflow.start_run():
    prepared = prepare_data(selected_feature)
    train_and_evaluate(prepared, categorical_feature)
