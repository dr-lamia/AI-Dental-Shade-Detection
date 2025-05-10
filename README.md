
# 🦷 AI Dental Shade Detection – Streamlit App

A streamlined, AI-powered dental web application for detecting and comparing tooth shades using the CIE-LAB color space and VITA mapping.

Built with [Streamlit](https://streamlit.io) for instant deployment and easy use by dental professionals and researchers.

---

## 🎯 Features

- ✅ Upload dental images (JPG/PNG)
- ✅ Extract CIE-LAB shade values
- ✅ Match closest **VITA Classic** shade
- ✅ Calculate ΔE (color difference) for comparison
- ✅ Save previous results for monitoring shade stability
- ✅ Export professional **PDF report**

---

## 🚀 Run Locally

### 1. Clone this repo
```bash
git clone https://github.com/yourusername/streamlit-ai-dental-app.git
cd streamlit-ai-dental-app
````

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to your GitHub account
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and select this repo
4. Hit **Deploy**!

---

## 🧪 Technologies Used

* `streamlit` – Interactive web UI
* `Pillow` – Image loading
* `scikit-image` – CIE-LAB conversion
* `NumPy` – ΔE calculations
* `ReportLab` – PDF report generation

---

## 📄 License

MIT License © 2025 Dr. Lamiaa El-Fadaly
Pioneer in AI in Dentistry 🧠🦷

