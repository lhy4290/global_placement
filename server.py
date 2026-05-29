from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import requests  # ✨ 用於呼叫 Gemini API
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import gaussian_kde
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'salary_model.pkl'
DATA_PATH = 'placed_students_salary.csv'  # 🎯 清洗後的正確資料集

# 載入 .env 檔案中的隱藏變數
load_dotenv()

# 安全地從環境變數中讀取金鑰
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ==========================================================
# 1. 智能記憶機制：模型持久化
# ==========================================================
def init_or_load_model():
    df_raw = pd.read_csv(DATA_PATH)
    df_placed = df_raw[df_raw['placement_status'] == 'Placed'].copy()
    
    features = ['cgpa', 'backlogs', 'college_tier', 'country', 'university_ranking_band', 
                'internship_count', 'aptitude_score', 'communication_score', 'specialization', 'industry', 'internship_quality_score']
    
    X_dummy = pd.get_dummies(df_placed[features], drop_first=True)
    model_columns = X_dummy.columns
    
    if os.path.exists(MODEL_PATH):
        print("💾 [機器記憶] 檢測到已存檔的模型，正在秒速載入隨機森林記憶...")
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        print("🧠 [大數據訓練] 未檢測到記憶檔，正在載入資料進行深度機器學習...")
        X = X_dummy
        y = df_placed['salary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ [記憶成功] 已將模型知識固化至本地檔案: {MODEL_PATH}")

    salaries = df_placed['salary'].values
    kde = gaussian_kde(salaries)
    x_curve = np.linspace(20000, 130000, 100)
    y_curve = kde(x_curve)
    distribution_data = [{'x': round(x, 2), 'y': round(y, 6)} for x, y in zip(x_curve, y_curve)]
    
    return df_placed, model, model_columns, distribution_data

df_placed, model, model_columns, distribution_data = init_or_load_model()

# ==========================================================
# 2. 預測路由 (整合隨機森林 + 台灣對照 + Gemini AI 職涯導師)
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
                
    predicted_salary = model.predict(input_df)[0]
    
    input_df_b = input_df.copy()
    input_df_b['internship_count'] = min(input_df['internship_count'].iloc[0] + 1, 5)
    sal_intern = model.predict(input_df_b)[0]
    
    input_df_c = input_df.copy()
    input_df_c['cgpa'] = min(input_df['cgpa'].iloc[0] + 1.0, 10.0)
    sal_cgpa = model.predict(input_df_c)[0]
    
    pr_value = (df_placed['salary'] < predicted_salary).mean() * 100
    
    # 台灣在地化與 PPP 計算
    ppp_multipliers = {'USA': 1.85, 'Germany': 1.45, 'UK': 1.55, 'Canada': 1.50, 'India': 0.45}
    target_country = user_input_dict.get('country', 'Germany')
    multiplier = ppp_multipliers.get(target_country, 1.0)
    
    usd_to_twd_rate = 32.5
    raw_twd_salary = predicted_salary * usd_to_twd_rate
    ppp_twd_salary = raw_twd_salary / multiplier
    
    taiwan_base_salaries = {'Data Science': 950000, 'AI/ML': 1100000, 'Cybersecurity': 850000, 'Core CS': 900000, 'Cloud': 880000}
    tw_spec = user_input_dict.get('specialization', 'Data Science')
    taiwan_estimated_base = taiwan_base_salaries.get(tw_spec, 900000)
    ppp_difference_pct = ((ppp_twd_salary - taiwan_estimated_base) / taiwan_estimated_base) * 100

    # ----------------------------------------------------------
    # ✨ 調用 Google Gemini API 產生客製化職涯健檢報告
    # ----------------------------------------------------------
    ai_advice = "（AI 導師開小差了，請確認 API Key 是否設定正確）"
    
    if GEMINI_API_KEY:
        try:
            prompt = f"""
            你是一位專業的跨國科技獵頭與職涯發展專家。請根據以下學生的條件，給出一份精簡、具建設性的個人化履歷健檢與跨國求職策略診斷。

            【學生簡歷背景】
            - 學業成績 (CGPA): {user_input_dict.get('cgpa')} / 10.0
            - 學校階層: {user_input_dict.get('college_tier')} (排名區間: {user_input_dict.get('university_ranking_band')})
            - 專業領域: {user_input_dict.get('specialization')}
            - 累計實習次數: {user_input_dict.get('internship_count')} 次 (實習質量分: {user_input_dict.get('internship_quality_score')} / 10.0)
            - 能力測驗: 性向測驗 {user_input_dict.get('aptitude_score')} 分 / 溝通能力 {user_input_dict.get('communication_score')} 分
            - 目標求職國家: {target_country} (目標行業: {user_input_dict.get('industry')})
            - 大數據預估海外年薪: ${round(predicted_salary):,} USD (市場超越度: PR {int(pr_value)})

            請分段提供以下三點回覆（總字數控制在 250 字內，口吻要專業、積極、簡潔）：
            1. 【優勢評估】：一句話點出他簡歷中最具競爭力的亮點。
            2. 【弱點診斷】：指出他若想挑戰更高薪資，目前最大的瓶頸是什麼。
            3. 【行動方針】：給予一個具體可執行的優化建議。
            """
            
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            headers = {'Content-Type': 'application/json'}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                res_json = response.json()
                ai_advice = res_json['candidates'][0]['content']['parts'][0]['text']
                ai_advice = ai_advice.replace('\n', '<br>')
                ai_advice = ai_advice.replace('**', '')
            else:
                print(f"❌ Gemini API 回傳錯誤碼: {response.status_code}, 回應內容: {response.text}")
        except Exception as e:
            print(f"❌ Gemini API 呼叫失敗: {str(e)}")
            ai_advice = "【系統提示】目前 AI 導師連線忙碌中，請稍後再試。但下方的常態大數據圖表與預測仍完全正常運作！"

    return jsonify({
        'status': 'success',
        'predicted_salary': round(predicted_salary),
        'pr_value': round(pr_value, 1),
        'dist_curve': distribution_data, 
        'bar_chart_data': {
            'labels': ['1. 維持履歷現狀', '2. 多增加 1 次實習經驗', '3. 學業成績 (CGPA) 提升 1.0'],
            'values': [round(predicted_salary), round(sal_intern), round(sal_cgpa)],
            'increases': [0, round(sal_intern - predicted_salary), round(sal_cgpa - predicted_salary)]
        },
        'target_country_name': target_country,
        'ppp_twd_salary': round(ppp_twd_salary),
        'taiwan_estimated_base': taiwan_estimated_base,
        'ppp_difference_pct': round(ppp_difference_pct, 1),
        'ai_advice': ai_advice
    })

# ==========================================================
# 3. 反向探索路由
# ==========================================================
@app.route('/reverse_search', methods=['POST'])
def reverse_search():
    request_data = request.json
    target_salary = float(request_data.get('target_salary', 70000))
    
    lower_bound = target_salary * 0.85
    upper_bound = target_salary * 1.15
    df_match = df_placed[(df_placed['salary'] >= lower_bound) & (df_placed['salary'] <= upper_bound)]
    
    if len(df_match) < 10:
        df_placed['salary_diff'] = (df_placed['salary'] - target_salary).abs()
        df_match = df_placed.nsmallest(50, 'salary_diff')
    
    avg_cgpa = df_match['cgpa'].mean()
    avg_internships = df_match['internship_count'].mean()
    avg_internship_quality = df_match['internship_quality_score'].mean()
    avg_communication = df_match['communication_score'].mean()
    avg_aptitude = df_match['aptitude_score'].mean()
    
    top_country = df_match['country'].mode()[0] if not df_match['country'].empty else "USA"
    top_specialization = df_match['specialization'].mode()[0] if not df_match['specialization'].empty else "Data Science"
    top_industry = df_match['industry'].mode()[0] if not df_match['industry'].empty else "Tech"
    
    return jsonify({
        'status': 'success',
        'count': len(df_match),
        'avg_cgpa': round(avg_cgpa, 2),
        'avg_internships': round(avg_internships, 1),
        'avg_internship_quality': round(avg_internship_quality, 1),
        'avg_communication': round(avg_communication, 1),
        'avg_aptitude': round(avg_aptitude, 1),
        'top_country': top_country,
        'top_specialization': top_specialization,
        'top_industry': top_industry
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)