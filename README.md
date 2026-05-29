# 跨國就業薪資智能評估系統 (Global Salary Predictor)

本專案專為 Windows 使用者優化設計。透過以下簡單的步驟，您即可在本機電腦上完整復刻、並透過現代化網頁介面執行這套結合「隨機森林機器學習」與「大數據職涯導師」的完全體系統。

---

## 快速開始與環境配置 (Quick Start)

請按照以下三個簡單步驟進行配置與啟動：

### 步驟一：下載本專案至您的電腦
點擊本 GitHub 倉庫右上角的 **綠色 `Code` 按鈕**，並選擇 **`Download ZIP`**。下載完成後，將壓縮檔解壓縮至您的桌面或任意工作資料夾。

### 步驟二：建立 Python 虛擬環境與安裝套件
本專案之預測引擎與統計分析需要特定的資料科學套件支援。請打開您的核心編輯器（推薦使用 VS Code）的終端機，確保路徑處於專案根目錄下，並依序執行以下指令：

```bash
# 1. 建立專屬虛擬環境 (.venv)
py -m venv .venv

# 2. 升級 pip 並安裝本專案必備之網頁與機器學習套件
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 步驟三：啟動系統（後端運行 + 前端 Live Server 渲染）
由於現代瀏覽器的嚴格安全機制（CORS 跨域限制），直接雙擊開啟 HTML 檔案會導致前端無法順利向後端請求 Gemini AI 報告。請務必同時啟動後端與本地網頁伺服器：

1. 啟動 Flask 後端預測引擎
在 VS Code 終端機中，執行以下指令叫醒後端伺服器：

```bash
.\.venv\Scripts\python.exe server.py
```

系統將自動在背景啟動 Flask 服務（進行隨機森林模型的智能記憶讀取/初次訓練），當看到 `* Running on http://127.0.0.1:5000` 即代表後端通暢。
