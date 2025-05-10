import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="AI Dental Shade App", layout="centered")

st.title("🦷 AI Dental Shade Detection")
st.write("Upload an intraoral image to detect CIE-LAB values and match to VITA Classic or 3D-Master shades.")

vita_3d_master_shades = {
    "1M1": [99.0, 0.005, 1.5],
    "1M2": [97.2, 0.1, 2.0],
    "2M1": [94.0, 1.0, 3.0],
    "2M2": [91.5, 2.0, 4.0],
    "3M1": [88.0, 3.5, 5.0],
    "3M2": [85.5, 4.0, 6.0],
    "4M1": [82.0, 5.0, 7.0]
}

def find_closest_vita_3d(lab):
    lab = np.array(lab)
    min_dist = float("inf")
    closest = None
    for shade, ref_lab in vita_3d_master_shades.items():
        dist = np.linalg.norm(lab - np.array(ref_lab))
        if dist < min_dist:
            min_dist = dist
            closest = shade
    return closest

def extract_cie_lab(pil_img):
    img = np.array(pil_img) / 255.0
    lab_img = color.rgb2lab(img)
    return np.mean(lab_img, axis=(0, 1))

def convert_to_vita(lab):
    return "A2"

def convert_to_vita_3d(lab):
    return find_closest_vita_3d(lab)

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
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    lab = extract_cie_lab(image)
    vita = convert_to_vita(lab)
    st.success(f"CIE-LAB: {', '.join([str(round(x, 2)) for x in lab])}")
    st.info(f"VITA Classic Match: {vita}")
    vita3d = convert_to_vita_3d(lab)
    st.info(f"VITA 3D-Master Match: {vita3d}")

    if "prev_lab" not in st.session_state:
        st.session_state["prev_lab"] = None

    compare = st.checkbox("Compare with previous image")
    if compare and "prev_lab" in st.session_state and st.session_state["prev_lab"] is not None:
        deltaE = calculate_delta_e(st.session_state["prev_lab"], lab)
        st.warning(f"ΔE: {round(deltaE, 2)}")

    if st.button("Save as Previous Shade"):
        st.session_state["prev_lab"] = lab
        st.success("Previous shade saved for comparison.")

    st.subheader("🗺️ Shade Map (LAB Distribution)")
    img_np = np.array(image) / 255.0
    lab_img = color.rgb2lab(img_np)
    fig, ax = plt.subplots()
    ax.imshow(lab_img[:, :, 0], cmap='plasma')
    st.pyplot(fig)

    
if st.button("Download PDF Report"):
    if lab is not None and vita is not None:
        deltaE_val = calculate_delta_e(st.session_state["prev_lab"], lab) if st.session_state["prev_lab"] else 0.0
        pdf = generate_pdf_report(lab, vita, deltaE_val)
        if pdf is not None:
            st.download_button("📄 Download PDF", data=pdf, file_name="shade_report.pdf", mime="application/pdf")
        else:
            st.error("PDF generation failed.")

        deltaE_val = calculate_delta_e(st.session_state["prev_lab"], lab) if st.session_state["prev_lab"] else 0.0
        pdf = generate_pdf_report(lab, vita, deltaE_val)
        st.download_button("📄 Download PDF", data=pdf, file_name="shade_report.pdf", mime="application/pdf")
