from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'salary_model.pkl'
DATA_PATH = 'global_placement.csv'

# ==========================================================
# 1. 智能記憶機制：模型持久化 (Model Persistence)
# ==========================================================
def init_or_load_model():
    df_raw = pd.read_csv(DATA_PATH)
    df_placed = df_raw[df_raw['placement_status'] == 'Placed'].copy()
    
    features = ['cgpa', 'backlogs', 'college_tier', 'country', 'university_ranking_band', 
                'internship_count', 'aptitude_score', 'communication_score', 'specialization', 'industry', 'internship_quality_score']
    
    # 為了保持 One-Hot Encoding 的欄位結構一致，我們需要拿 dummy 欄位名
    X_dummy = pd.get_dummies(df_placed[features], drop_first=True)
    model_columns = X_dummy.columns
    
    # 檢查是否已經有機器記憶檔案 (.pkl)
    if os.path.exists(MODEL_PATH):
        print("💾 [機器記憶] 檢測到已存檔的模型，正在秒速載入隨機森林記憶...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        print("🧠 [大數據訓練] 未檢測到記憶檔，正在載入 10,000 筆資料進行深度機器學習...")
        X = X_dummy
        y = df_placed['salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 🌟 讓機器記住模型：將訓練好的大腦存成 pkl 檔
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ [記憶成功] 已將模型知識固化至本地檔案: {MODEL_PATH}")

    # 預先計算分佈曲線數據 (維持不變)
    salaries = df_placed['salary'].values
    kde = gaussian_kde(salaries)
    x_curve = np.linspace(20000, 130000, 100)
    y_curve = kde(x_curve)
    distribution_data = [{'x': round(x, 2), 'y': round(y, 6)} for x, y in zip(x_curve, y_curve)]
    
    return df_placed, model, model_columns, distribution_data

# 初始化獲取核心組件
df_placed, model, model_columns, distribution_data = init_or_load_model()

# ==========================================================
# 2. 預測路由 (API Endpoint)
# ==========================================================
@app.route('/predict', methods=['POST'])
def predict():
    user_input_dict = request.json
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    for col in ['cgpa', 'backlogs', 'internship_count', 'aptitude_score', 'communication_score', 'internship_quality_score']:
        if col in user_input_dict:
            input_df[col] = float(user_input_dict[col])
            
    for col in ['college_tier', 'country', 'university_ranking_band', 'specialization', 'industry']:
        if col in user_input_dict:
            dummy_col = f"{col}_{user_input_dict[col]}"
            if dummy_col in input_df.columns:
                input_df[dummy_col] = 1
                
    # 利用固化好的隨機森林進行即時特徵推論
    predicted_salary = model.predict(input_df)[0]
    
    # What-If 敏感度情境模擬
    input_df_b = input_df.copy()
    input_df_b['internship_count'] = min(input_df['internship_count'].iloc[0] + 1, 5)
    sal_intern = model.predict(input_df_b)[0]
    
    input_df_c = input_df.copy()
    input_df_c['cgpa'] = min(input_df['cgpa'].iloc[0] + 1.0, 10.0)
    sal_cgpa = model.predict(input_df_c)[0]
    
    pr_value = (df_placed['salary'] < predicted_salary).mean() * 100
    
    return jsonify({
        'status': 'success',
        'predicted_salary': round(predicted_salary),
        'pr_value': round(pr_value, 1),
        'dist_curve': distribution_data, 
        'bar_chart_data': {
            'labels': ['1. 維持履歷現狀', '2. 多增加 1 次實習經驗', '3. 學業成績 (CGPA) 提升 1.0'],
            'values': [round(predicted_salary), round(sal_intern), round(sal_cgpa)],
            'increases': [0, round(sal_intern - predicted_salary), round(sal_cgpa - predicted_salary)]
        }
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)