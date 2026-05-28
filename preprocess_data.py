import pandas as pd
import re

def clean_salary(row):
    salary_str = row['薪資原始字串']
    title_str = str(row['職稱'])
    edu_str = str(row['職缺所需學歷'])

    if pd.isna(salary_str):
        return None, None
    
    salary_str = str(salary_str)

    if "時薪" in salary_str or "實習" in title_str:
        return None, None

    if "待遇面議" in salary_str:
        is_senior = any(k in title_str.lower() for k in ['資深', 'senior', '高級', 'architect', '架構', '主管'])
        has_tech = any(k in title_str.lower() for k in ['生成式', 'llm', 'generative', '深度學習', 'deep learning'])
        is_high_edu = any(k in edu_str for k in ['碩士', '博士'])

        if is_senior and has_tech:
            return 65000, 85000
        elif is_senior or (has_tech and is_high_edu):
            return 55000, 75000
        elif is_high_edu:
            return 48000, 60000
        else:
            return 40000, 50000
    
    clean_str = salary_str.replace(',', '')
    numbers = re.findall(r'\d+', clean_str)
    numbers = [int(n) for n in numbers]

    if not numbers:
        return None, None
    
    raw_min = numbers[0]
    raw_max = numbers[1] if len(numbers) > 1 else numbers[0]

    if "年薪" in salary_str:
        return round(raw_min / 12), round(raw_max / 12)
    
    return raw_min, raw_max

def extract_title_features(df):
    df['is_senior'] = df['職稱'].str.contains('資深|Senior|高級|Architect|架構', case=False).astype(int)
    df['has_hot_tech'] = df['職稱'].str.contains('生成式|LLM|Generative|Algorithm|演算法|深度學習|Deep Learning', case=False).astype(int)

    df['is_data_role'] = df['職稱'].str.contains('資料|Data|分析', case=False).astype(int)
    df['is_algo_role'] = df['職稱'].str.contains('演算法|Algorithm|模型|研究', case=False).astype(int)
    df['is_sys_role'] = df['職稱'].str.contains('系統|System|軟體|Software|IT|Infra', case=False).astype(int)

    return df

def preprocess():
    print("開始數據預處理")
    df = pd.read_csv("ai_jobs_raw.csv")
    print(f"成功讀入原始數據共：{len(df)} 筆")

    salary_res = df.apply(clean_salary, axis=1)
    df['salary_min'] = [x[0] if x is not None else None for x in salary_res]
    df['salary_max'] = [x[1] if x is not None else None for x in salary_res]

    df = df.dropna(subset=['salary_min'])
    df = df[df['salary_min'] >= 25000]
    print(f"有效薪資數據：{len(df)} 筆")

    edu_map = {
        "學歷不拘": 0, "不拘": 0, 
        "高中以下": 1, "高中": 1,
        "專科": 2, 
        "大學": 3,"大學建設": 3, 
        "碩士": 4, 
        "博士": 5
    }
    df['edu_level'] = df['職缺所需學歷'].map(edu_map).fillna(0).astype(int)

    initial_count = len(df)
    df = df.drop_duplicates(subset=['職稱', 'salary_min', 'salary_max'])
    print(f"已移除 {initial_count - len(df)} 筆重複數據")

    df.to_csv("ai_jobs_cleaned.csv", index=False, encoding="utf-8-sig")
    print("清洗完成")

    print("開始特徵工程")
    df = extract_title_features(df)

    df.to_csv("ai_jobs_final.csv", index=False, encoding="utf-8-sig")
    print(f"預處理和特徵工程已完成。最終可用於訓練的有效數據：{len(df)} 筆")
    print(df[['職稱', 'salary_min', 'edu_level', 'is_senior', 'has_hot_tech']].head())

if __name__ == "__main__":
    preprocess()