import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')

baseline_features = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'EXT_SOURCE_1']

X = train[baseline_features].fillna(0)
y = train['TARGET']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)

y_val_pred = baseline_model.predict_proba(X_val)[:,1]
print("Baseline Validation AUC:", roc_auc_score(y_val, y_val_pred))

X_test = test[baseline_features].fillna(0)
baseline_submission = pd.DataFrame({
    'SK_ID_CURR': test['SK_ID_CURR'],
    'TARGET': baseline_model.predict_proba(X_test)[:,1]
})
baseline_submission.to_csv('baseline_submission.csv', index=False)

def preprocess_data(df, features):
    df = df.copy()
    numeric_feats = df[features].select_dtypes(include=np.number).columns
    df[numeric_feats] = df[numeric_feats].fillna(df[numeric_feats].mean())
    cat_feats = df[features].select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=cat_feats, dummy_na=True)
    return df

train_fe1 = train.copy()
test_fe1 = test.copy()
train_fe1['CREDIT_INCOME_RATIO'] = train_fe1['AMT_CREDIT'] / train_fe1['AMT_INCOME_TOTAL']
test_fe1['CREDIT_INCOME_RATIO'] = test_fe1['AMT_CREDIT'] / test_fe1['AMT_INCOME_TOTAL']
patterns = [['AMT_INCOME_TOTAL','DAYS_BIRTH','EXT_SOURCE_1','CREDIT_INCOME_RATIO']]

train_fe2 = train.copy()
test_fe2 = test.copy()
train_fe2['LOG_INCOME'] = np.log1p(train_fe2['AMT_INCOME_TOTAL'])
test_fe2['LOG_INCOME'] = np.log1p(test_fe2['AMT_INCOME_TOTAL'])
patterns.append(['AMT_INCOME_TOTAL','DAYS_BIRTH','EXT_SOURCE_1','LOG_INCOME'])

train_fe3 = train.copy()
test_fe3 = test.copy()
train_fe3['EXT_SOURCES_PROD'] = train_fe3['EXT_SOURCE_1'] * train_fe3['EXT_SOURCE_2']
test_fe3['EXT_SOURCES_PROD'] = test_fe3['EXT_SOURCE_1'] * test_fe3['EXT_SOURCE_2']
patterns.append(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCES_PROD','AMT_INCOME_TOTAL'])

train_fe4 = train.copy()
test_fe4 = test.copy()
train_fe4['AGE_YEARS'] = (-train_fe4['DAYS_BIRTH']) // 365
test_fe4['AGE_YEARS'] = (-test_fe4['DAYS_BIRTH']) // 365
train_fe4['AGE_BIN'] = pd.cut(train_fe4['AGE_YEARS'], bins=5, labels=False)
test_fe4['AGE_BIN'] = pd.cut(test_fe4['AGE_YEARS'], bins=5, labels=False)
patterns.append(['AGE_YEARS','AGE_BIN','EXT_SOURCE_1','AMT_CREDIT'])

train_fe5 = train.copy()
test_fe5 = test.copy()
features = ['AMT_INCOME_TOTAL','AMT_CREDIT','DAYS_BIRTH','EXT_SOURCE_1','NAME_CONTRACT_TYPE']
X_train_fe5 = preprocess_data(train_fe5, features)
y_train_fe5 = train_fe5['TARGET']
X_val_fe5 = X_train_fe5.sample(frac=0.2, random_state=42)
y_val_fe5 = y_train_fe5.loc[X_val_fe5.index]
X_train_fe5 = X_train_fe5.drop(X_val_fe5.index)

def train_and_validate(train_df, test_df, features, y_train, pattern_name):
    X = train_df[features].fillna(0)
    y = y_train
    X_train, X_val, y_train_split, y_val_split = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_split)
    y_val_pred = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val_split, y_val_pred)
    print(f'{pattern_name} Validation AUC:', auc)
    X_test = test_df[features].fillna(0)
    submission = pd.DataFrame({
        'SK_ID_CURR': test_df['SK_ID_CURR'],
        'TARGET': model.predict_proba(X_test)[:,1]
    })
    submission.to_csv(f'{pattern_name}_submission.csv', index=False)

train_and_validate(train_fe1, test_fe1, patterns[0], train['TARGET'], 'Pattern1')
train_and_validate(train_fe2, test_fe2, patterns[1], train['TARGET'], 'Pattern2')
train_and_validate(train_fe3, test_fe3, patterns[2], train['TARGET'], 'Pattern3')
train_and_validate(train_fe4, test_fe4, patterns[3], train['TARGET'], 'Pattern4')

model_fe5 = RandomForestClassifier(n_estimators=100, random_state=42)
model_fe5.fit(X_train_fe5, y_train_fe5)
y_val_pred5 = model_fe5.predict_proba(X_val_fe5)[:,1]
print('Pattern5 Validation AUC:', roc_auc_score(y_val_fe5, y_val_pred5))
submission_fe5 = pd.DataFrame({
    'SK_ID_CURR': test_fe5['SK_ID_CURR'],
    'TARGET': model_fe5.predict_proba(preprocess_data(test_fe5, features))[:,1]
})
submission_fe5.to_csv('Pattern5_submission.csv', index=False)

print("All submissions saved. Ready to upload to Kaggle!")
