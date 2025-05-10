
import streamlit as st
from PIL import Image
from skimage import color
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Dental Shade App", layout="centered")

st.title("🦷 AI Dental Shade Detection")
st.write("Upload a dental image to detect CIE-LAB values and match to VITA shade.")

def extract_cie_lab(pil_img):
    img = np.array(pil_img) / 255.0
    lab_img = color.rgb2lab(img)
    return np.mean(lab_img, axis=(0, 1))

def convert_to_vita(lab):
    # Placeholder logic for mapping
    return "A2"

def calculate_delta_e(lab1, lab2):
    return float(np.linalg.norm(np.array(lab1) - np.array(lab2)))

def generate_pdf_report(lab, vita, deltaE):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(100, 800, "Dental Shade Report")
    c.drawString(100, 780, f"VITA Shade: {vita}")
    c.drawString(100, 760, f"CIE-LAB: {', '.join([str(round(l, 2)) for l in lab])}")
    c.drawString(100, 740, f"Delta E: {deltaE}")
    c.save()
    buffer.seek(0)
    return buffer

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    lab = extract_cie_lab(image)
    vita = convert_to_vita(lab)

    st.success(f"CIE-LAB: {', '.join([str(round(x, 2)) for x in lab])}")
    st.info(f"VITA Match: {vita}")

    if "prev_lab" not in st.session_state:
        st.session_state.prev_lab = None

    compare = st.checkbox("Compare with previous image")
    if compare and st.session_state.prev_lab:
        deltaE = calculate_delta_e(st.session_state.prev_lab, lab)
        st.warning(f"ΔE: {round(deltaE, 2)}")

    if st.button("Save as Previous Shade"):
        st.session_state.prev_lab = lab
        st.success("Previous shade saved for comparison.")

    if st.button("Download PDF Report"):
        deltaE_val = calculate_delta_e(st.session_state.prev_lab, lab) if st.session_state.prev_lab else 0.0
        pdf = generate_pdf_report(lab, vita, deltaE_val)
        st.download_button("📄 Download PDF", data=pdf, file_name="shade_report.pdf", mime="application/pdf")
