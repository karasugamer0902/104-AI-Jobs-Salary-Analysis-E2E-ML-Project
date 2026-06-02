from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="104 AI 工程師起薪預測系統 API",
    description="結合機器學習隨機森林模型與特徵工程，預測全職 AI 相關職缺的起薪",
    version="1.0.0"
)

model_filename = "ai_salary_model.pkl"

def find_and_load_model(start_dir: Path, filename: str):
    current_dir = start_dir.resolve()

    while True:
        print(f"正在對準目錄層級: {current_dir}")

        direct_path = current_dir / filename
        if direct_path.is_file():
            print(f"成功找到目標: {direct_path}")
            return joblib.load(direct_path)
        
        try:
            for sub_path in current_dir.glob(f"**/{filename}"):
                if sub_path.is_file():
                    print(f"成功找到目標: {sub_path}")
                    return joblib.load(sub_path)
        except PermissionError:
            pass

        if current_dir.parent == current_dir:
            break

        current_dir = current_dir.parent
    
    raise FileNotFoundError(f"找不到 {filename}")

try:
    app_dir = Path(__file__).parent
    artifacts = find_and_load_model(app_dir, model_filename)
    if artifacts is not None:
        model = artifacts['model']
        features_list = artifacts.get('features', [])
        print("成功載入模型")
    else:
        model = None
except Exception as e:
    print(f"模型載入失敗，錯誤原因：{e}")
    model = None

class JobPredictionInput(BaseModel):
    edu_level: int = Field(..., description="學歷等級 (0:不拘, 1:高中, 2:專科, 3:大學, 4:碩士, 5:博士)", ge=0, le=5)
    is_senior: int = Field(..., description="是否為資深/高級/管理職缺 (0:否, 1:是)", ge=0, le=1)
    has_hot_tech: int = Field(..., description="是否要求熱門 AI 技術例如 LLM / 生成式 (0:否, 1:是)", ge=0, le=1)
    is_data_role: int = Field(..., description="是否為數據分析 / Data 相關角色 (0:否, 1:是)", ge=0, le=1)
    is_algo_role: int = Field(..., description="是否為演算法 / 研究相關角色 (0:否, 1:是)", ge=0, le=1)
    is_sys_role: int = Field(..., description="是否為系統 / 軟體 / Infra 相關角色 (0:否, 1:是)", ge=0, le=1)

@app.post("/predict", summary="預測 AI 職缺月薪起薪")
def predict_salary(payload: JobPredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="模型未成功載入，無法執行預測")
    
    input_data = pd.DataFrame([{
        'edu_level': payload.edu_level,
        'is_senior': payload.is_senior,
        'has_hot_tech': payload.has_hot_tech,
        'is_data_role': payload.is_data_role,
        'is_algo_role': payload.is_algo_role,
        'is_sys_role': payload.is_sys_role
    }])

    try:
        predicted_log = model.predict(input_data)
        predicted_real_salary = np.expm1(predicted_log)[0]

        return {
            "status": "success",
            "prediction": {
                "estimated_min_salary_ntd": round(predicted_real_salary),
                "formatted_salary": f"$ {round(predicted_real_salary):,}NTD"
            },
            "input_features_echo": payload.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"預測過程中發生錯誤:{str(e)}")
                            
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}