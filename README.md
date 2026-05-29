# 跨國就業薪資智能評估系統 (Global Salary Predictor)

本專案專為 Windows 使用者優化設計。透過以下簡單的步驟，您即可在本機電腦上完整復刻、並透過現代化網頁介面執行這套結合「隨機森林機器學習」與「大數據職涯導師」的完全體系統。

---

## 快速開始與環境配置 (Quick Start)

請按照以下三個簡單步驟進行配置與啟動：

### 步驟一：下載本專案至您的電腦
點擊本 GitHub 倉庫右上角的 綠色 `Code` 按鈕，並選擇 `Download ZIP`。下載完成後，將壓縮檔解壓縮至您的桌面或任意工作資料夾。

### 步驟二：建立 Python 虛擬環境與安裝套件
本專案之預測引擎與統計分析需要特定的資料科學套件支援。請打開您的核心編輯器（推薦使用 VS Code）的終端機，確保路徑處於專案根目錄下，並依序執行以下指令：

```bash
# 1. 建立專屬虛擬環境 (.venv)
py -m venv .venv

# 2. 升級 pip 並安裝本專案必備之網頁與機器學習套件
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. 輸入自己的金鑰
請將 .env.example 複製一份並改名為 .env，然後填入你自己的金鑰。
```

### 步驟三：啟動系統（後端運行 + 前端 Live Server 渲染）
由於現代瀏覽器的嚴格安全機制（CORS 跨域限制），直接雙擊開啟 HTML 檔案會導致前端無法順利向後端請求 Gemini AI 報告。請務必同時啟動後端與本地網頁伺服器：

1. 啟動 Flask 後端預測引擎
在 VS Code 終端機中，執行以下指令叫醒後端伺服器：

```bash
.\.venv\Scripts\python.exe server.py
```

系統將自動在背景啟動 Flask 服務（進行隨機森林模型的智能記憶讀取/初次訓練），當看到 `* Running on http://127.0.0.1:5000`即代表後端通暢。

2. 透過 VS Code 「Live Server」開啟前端網頁
為了打通連線大動脈，我們需要讓網頁跑在本地的`http://`協定環境下：

進入 VS Code 左側工具列的 「延伸模組 (Extensions)」（快速鍵`Ctrl + Shift + X`）。

搜尋`Live Server`（作者為 Ritwick Dey）並點擊 安裝 (Install)。

安裝完成後，回到檔案總管點開`index.html`。

在程式碼任意空白處點擊滑鼠右鍵，選擇`Open with Live Server`（或點擊 VS Code 右下角狀態列的`Go Live`）。

瀏覽器會自動彈出網址為`http://127.0.0.1:5500/index.html`的現代化卡片網頁，您即可自由調整參數，享受 Chart.js 動態互動圖表與 Gemini AI 帶來的智能職涯診斷報告！

### 📂 核心專案結構 (Project Architecture)

```bash
├── server.py                 # 基於 Flask 框架的後端預測引擎，負責呼叫機器記憶進行特徵推論、台灣 PPP 換算與 API 串接。
├── index.html                # 前端使用者介面，整合現代化 CSS 卡片式排版與 Chart.js 客戶端動態互動繪圖技術。
├── placed_students_salary.csv # 核心大數據資料庫（已清洗），供機器學習模型第一次啟動時深度學習與常態分佈計算使用。
├── salary_model.pkl          # 模型持久化檔案（記憶體固化），生成後可供系統秒速載入，免去重複訓練時間。
└── requirements.txt          # 本專案必備套件清單（包含 Flask, Flask-CORS, Scikit-Learn, SciPy, Requests 等）。
```

### ✨ 核心系統亮點 (Key Features)
隨機森林特徵模擬：即時動態權重計算，精準預估海外年薪落點與就業市場超越度 (PR 值)。

夢想薪資逆向工程：反向探索導航機制，輸入理想薪資即可由萬筆數據自動推算達標之學長姐簡歷特徵。

生活素質實質平價 (PPP) 精算：依據世界銀行購買力平價指數，精算扣除當地物價權重後的台幣實質感受年薪，理性對決留台與跨國發展。

Gemini 正式版 API 聯名診斷：完美對接雲端大語言模型，針對個人學歷、實習質量、溝通性向指標進行定性之優勢評估與具體行動方針指引。
