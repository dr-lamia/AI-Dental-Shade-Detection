# 🦷 AI Dental Shade Detection App

An AI-powered Streamlit web app for detecting, comparing, and documenting dental shades using intraoral photographs. Built for both clinical and educational use.

---

## 🚀 Features (Latest)

- 📤 Upload intraoral images (JPG/PNG)
- 🎯 Extract average CIE-LAB values
- 🎨 Match closest **VITA Classic** and **VITA 3D-Master** shade
- 🧮 Calculate **ΔE (color difference)** for stability tracking
- 🗺️ Visualize **shade heatmaps (L* channel)**
- 📄 **Generate PDF reports** with shade, LAB, and ΔE info
- ❌ Webcam capture disabled for Streamlit Cloud compatibility

---

## 🧑‍🎓 Educational Use

- Compare shades before/after bleaching
- Test camera/light influence on shade selection
- Train students in objective esthetic assessment
- Export reports for documentation or presentations

---

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

For free deployment, use [Streamlit Cloud](https://share.streamlit.io):

1. Push this repo to GitHub
2. Connect GitHub to Streamlit Cloud
3. Select `app.py` as the main file
4. Deploy and share the public link

---

## 🧪 Tech Stack

- `streamlit`
- `scikit-image`
- `numpy`
- `reportlab`
- `matplotlib`
- `pillow`

---

## 👩‍⚕️ Author

Dr. Lamiaa El-Fadaly  
Pioneer in AI in Dentistry | 2025

---

## 📄 License

MIT License
