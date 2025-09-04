import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

data = pd.read_csv("train.csv")
X = data[["GrLivArea", "YearBuilt"]]
y = data["SalePrice"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = LinearRegression().fit(X_train_scaled, y_train)
svm = SVR(kernel="rbf", C=100).fit(X_train_scaled, y_train)
tree = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)

pred_lr = lr.predict(X_val_scaled)
pred_svm = svm.predict(X_val_scaled)
pred_tree = tree.predict(X_val)

mse_lr = mean_squared_error(y_val, pred_lr)
mse_svm = mean_squared_error(y_val, pred_svm)
mse_tree = mean_squared_error(y_val, pred_tree)

pred_blend = (pred_lr + pred_svm + pred_tree) / 3
mse_blend = mean_squared_error(y_val, pred_blend)

def bagging(model, X_train, y_train, X_val, n_estimators=10):
    preds = []
    for i in range(n_estimators):
        X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=i)
        m = model.fit(X_boot, y_boot)
        preds.append(m.predict(X_val))
    return np.mean(preds, axis=0)

bagged_preds = bagging(DecisionTreeRegressor(max_depth=5), X_train, y_train, X_val)
mse_bagging = mean_squared_error(y_val, bagged_preds)

base_models = [
    LinearRegression(),
    SVR(kernel="rbf", C=100),
    DecisionTreeRegressor(max_depth=5)
]

blend_train = np.zeros((X_train.shape[0], len(base_models)))
blend_val = np.zeros((X_val.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train_scaled, y_train)
    blend_train[:, i] = model.predict(X_train_scaled)
    blend_val[:, i] = model.predict(X_val_scaled)

meta_model = LinearRegression()
meta_model.fit(blend_train, y_train)
stack_preds = meta_model.predict(blend_val)
mse_stacking = mean_squared_error(y_val, stack_preds)

results = {
    "Linear Regression": mse_lr,
    "SVM": mse_svm,
    "Decision Tree": mse_tree,
    "Blending": mse_blend,
    "Bagging": mse_bagging,
    "Stacking": mse_stacking
}

results_df = pd.DataFrame(results.items(), columns=["Model", "MSE"])
print(results_df)
