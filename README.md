# 📩 SMS Spam Detector – Feature Selection with Lasso

![Streamlit Logo](https://streamlit.io/images/brand/streamlit-mark-color.svg)

A **Streamlit app** that demonstrates **SMS spam detection** and **feature selection** using **Lasso Regression**. The project converts SMS messages into numerical features with **TF-IDF** and automatically identifies the **most important words** for classification.

Check out the live app here: **[🚀 Spam Detector App](https://spamdetectorfree.streamlit.app/)**

---

## ✨ Features

* 📂 Upload any **SMS spam dataset** in CSV format.
* 📝 Convert SMS messages into **TF-IDF features**.
* 🎯 Apply **Lasso Regression** for automatic **feature selection**.
* 🔧 Choose different **alpha values** to control selection strength.
* 📊 Display:

  * Total features
  * Non-zero features (selected)
  * Features eliminated
  * Percentage reduction
* 🌐 Fully interactive **browser-based app** with **Streamlit**.

---

## 🛠 Technologies Used

* 🐍 Python 3
* 🌟 Streamlit
* 📊 Pandas & Numpy
* 🧠 Scikit-learn (TF-IDF & Lasso Regression)

---

## 🚀 How to Use

1. Open the live app: [https://spamdetectorfree.streamlit.app/](https://spamdetectorfree.streamlit.app/)
2. Upload your **SMS spam dataset (CSV format)**.
3. Enter **Lasso alpha values** (e.g., `0.01,0.1,1`).
4. Click **Run Lasso Feature Selection** to see results interactively.

---

## 📂 Dataset

The default dataset used in this project:

* [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
* CSV should have **two columns**: `v1` (label: ham/spam) and `v2` (message text).

---

## 💻 Installation (Optional – Run Locally)

If you want to run the app on your machine:

```bash
# Clone the repo
git clone https://github.com/asiya260/spam-detector--.git
cd spam-detector--

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run sms_lasso_streamlit.py
```

---

## 👩‍💻 Author

**Asiya Maryam A** – BSc Computer Science (AI)

Check out the live app here: [🚀 Spam Detector App](https://spamdetectorfree.streamlit.app/)

---
