import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ==========================================================
# 網頁基本設定 (必須放在程式碼第一行)
# ==========================================================
st.set_page_config(page_title="跨國就業薪資智能評估工具", layout="wide")

# ==========================================================
# 1. 讀取資料與訓練模型 (加入快取避免每次重新整理都重訓，速度會極快)
# ==========================================================
@st.cache_data
def load_data_and_model():
    # ✨【已修正】確保函數內部讀取與使用的變數名稱一致
    df = pd.read_csv('placed_students_salary.csv')
    
    features = ['cgpa', 'backlogs', 'college_tier', 'country', 'university_ranking_band', 
                'internship_count', 'aptitude_score', 'communication_score', 'specialization', 'industry', 'internship_quality_score']
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return df, model, X.columns

# 外部接收傳回值時，再定義全域變數 df_placed 供後續圖表計算使用
df_placed, model, model_columns = load_data_and_model()

# ==========================================================
# 2. 預測工具核心函式
# ==========================================================
def predict_student_salary_tool(user_input_dict):
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    for col in ['cgpa', 'backlogs', 'internship_count', 'aptitude_score', 'communication_score', 'internship_quality_score']:
        if col in user_input_dict:
            input_df[col] = user_input_dict[col]
            
    for col in ['college_tier', 'country', 'university_ranking_band', 'specialization', 'industry']:
        if col in user_input_dict:
            dummy_col = f"{col}_{user_input_dict[col]}"
            if dummy_col in input_df.columns:
                input_df[dummy_col] = 1
                
    predicted_salary = model.predict(input_df)[0]
    pr_value = (df_placed['salary'] < predicted_salary).mean() * 100
    return predicted_salary, pr_value

# ==========================================================
# 3. 網格排版與網頁互動介面 (UI Widgets)
# ==========================================================
st.title("🎓 跨國就業薪資智能評估工具 (網頁預估模擬器)")
st.markdown("---")

# 將網頁分成左右兩欄
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏫 學術與背景資訊")
    cgpa_val = st.slider('學業成績 (CGPA):', min_value=4.0, max_value=10.0, value=7.8, step=0.1)
    backlogs_val = st.number_input('學科未過科數:', min_value=0, max_value=10, value=0, step=1)
    tier_val = st.selectbox('學校階層:', options=['Tier 1', 'Tier 2', 'Tier 3'], index=1)
    country_val = st.selectbox('目標就業國家:', options=['USA', 'Germany', 'UK', 'Canada', 'India'], index=1)
    ranking_val = st.selectbox('學校排名區間:', options=['Top 100', '100-300', '300+'], index=1)

with col2:
    st.subheader("💼 實務經驗與能力指標")
    intern_val = st.slider('目前實習次數:', min_value=0, max_value=5, value=1, step=1)
    quality_val = st.slider('實習質量分數:', min_value=1.0, max_value=10.0, value=6.0, step=0.1)
    aptitude_val = st.slider('性向測驗分數:', min_value=30.0, max_value=100.0, value=75.0, step=1.0)
    comm_val = st.slider('溝通能力分數:', min_value=30.0, max_value=100.0, value=70.0, step=1.0)
    spec_val = st.selectbox('專業領域:', options=['Data Science', 'AI/ML', 'Cybersecurity', 'Core CS', 'Cloud'], index=0)
    industry_val = st.selectbox('目標行業:', options=['Tech', 'Consulting', 'Healthcare', 'Finance', 'Manufacturing', 'Other'], index=0)

st.markdown("<br>", unsafe_allow_html=True)

# 建立開始評估按鈕
if st.button('🚀 開始評估潛在薪資', use_container_width=True):
    # 打包輸入值
    user_background = {
        'cgpa': cgpa_val, 'backlogs': backlogs_val, 'college_tier': tier_val,
        'country': country_val, 'university_ranking_band': ranking_val,
        'internship_count': intern_val, 'aptitude_score': aptitude_val,
        'communication_score': comm_val, 'specialization': spec_val,
        'industry': industry_val, 'internship_quality_score': quality_val
    }
    
    # 執行預測
    my_salary, my_pr = predict_student_salary_tool(user_background)
    
    # ==========================================================
    # 4. 繪製圖表與網頁輸出
    # ==========================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11))
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ----- [圖一：落點分佈] -----
    sns.kdeplot(df_placed['salary'], fill=True, color="skyblue", alpha=0.4, linewidth=2, ax=ax1, label='全體畢業生薪資分佈')
    ax1.axvline(x=my_salary, color='crimson', linestyle='--', linewidth=2.5)
    
    density_y = ax1.get_lines()[0].get_ydata()
    density_x = ax1.get_lines()[0].get_xdata()
    idx = (np.abs(density_x - my_salary)).argmin()
    ax1.plot(my_salary, density_y[idx], marker='*', color='gold', markersize=18, markeredgecolor='black', label='您的預測落點')
    
    mae_offset = 8000
    ax1.text(my_salary + 3000, density_y[idx] * 0.6, 
             f"【預測結果】\n估計年薪: ${my_salary:,.0f} USD\n預估範圍: ${my_salary-mae_offset:,.0f} ~ ${my_salary+mae_offset:,.0f}\n求職市場超越度: {my_pr:.1f}% (PR {int(my_pr)})", 
             fontsize=11, fontweight='bold', bbox=dict(facecolor='ivory', alpha=0.9, boxstyle="round,pad=0.5", edgecolor='crimson'))
    ax1.set_title('【工具圖一】個人背景在全體就業市場之薪資落點分佈圖', fontsize=14, fontweight='bold')
    ax1.set_xlabel('年薪範圍 (USD)', fontsize=11)
    ax1.set_ylabel('人數密度', fontsize=11)
    ax1.set_xlim(20000, 130000)
    ax1.legend(loc='upper left')
    
    # ----- [圖二：加薪模擬] -----
    scenarios = ['1. 維持現狀', '2. 多增加 1 次實習經驗', '3. 將學業成績(CGPA)提升 1.0']
    sal_1 = my_salary
        
    bg_more_intern = user_background.copy()
    bg_more_intern['internship_count'] = min(user_background['internship_count'] + 1, 5)
    sal_2, _ = predict_student_salary_tool(bg_more_intern)
        
    bg_better_cgpa = user_background.copy()
    bg_better_cgpa['cgpa'] = min(user_background['cgpa'] + 1.0, 10.0)
    sal_3, _ = predict_student_salary_tool(bg_better_cgpa)
        
    salary_results = [sal_1, sal_2, sal_3]
        
    # 改用單一色彩，避開新版 Seaborn 調色盤強度檢查
    sns.barplot(x=salary_results, y=scenarios, color='#3498db', width=0.4, ax=ax2)
        
    for i, val in enumerate(salary_results):
        increase = val - sal_1
        label_text = f"${val:,.0f} (↗ 加薪 ${increase:,.0f})" if increase > 0 else f"${val:,.0f} (起點基準)"
        ax2.text(val + 1500, i, label_text, va='center', fontweight='bold', fontsize=11)
            
    ax2.set_title('【工具圖二】不同履歷優化策略之「加薪模擬」對比圖', fontsize=14, fontweight='bold')
    ax2.set_xlabel('預估潛在年薪 (USD)', fontsize=11)
    ax2.set_xlim(20000, max(salary_results) * 1.3)
    
    plt.tight_layout()
    
    # 將生成的 Matplotlib 圖表直接線上渲染
    st.pyplot(fig)