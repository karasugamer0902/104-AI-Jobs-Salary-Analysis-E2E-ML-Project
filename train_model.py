import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_salary_predictor():
    print("載入特徵數據...")
    df = pd.read_csv("ai_jobs_final.csv")

    features = ['edu_level', 'is_senior', 'has_hot_tech', 'is_data_role', 'is_algo_role', 'is_sys_role']
    X = df[features]

    y_log = np.log1p(df['salary_min'])

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    print(f"訓練集樣本數：{len(X_train)}，測試集樣本數：{len(X_test)}")

    lr = LinearRegression()
    lr.fit(X_train, y_train_log)
    pred_lr_log = lr.predict(X_test)

    pred_lr = np.expm1(pred_lr_log)
    y_test_real = np.expm1(y_test_log)

    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train_log)
    pred_rf_log = rf.predict(X_test)
    pred_rf = np.expm1(pred_rf_log)

    print("\n評估模型結果")
    print(f"[線性回歸] MAE: $ {mean_absolute_error(y_test_real, pred_lr):,.0f} NTD | R² Score: {r2_score(y_test_real, pred_lr):.4f}")
    print(f"[隨機森林] MAE: $ {mean_absolute_error(y_test_real, pred_rf):,.0f} NTD | R² Score: {r2_score(y_test_real, pred_rf):.4f}")

    sample_data = pd.DataFrame([[3, 1, 1, 0, 1, 0]], columns=features)
    sample_pred_log = rf.predict(sample_data)
    print(f"\n測試預報（大學/資深/熱門AI技術/演算法）：$ {np.expm1(sample_pred_log)[0]:,.0f} NTD")

if __name__ == "__main__":
    train_salary_predictor()