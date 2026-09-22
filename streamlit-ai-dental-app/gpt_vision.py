from __future__ import annotations

import base64
import json
import re
from io import BytesIO

from PIL import Image

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


ALLOWED_3D_MASTER_SHADES = [
    "1M1", "1M2",
    "2L1.5", "2L2.5", "2M1", "2M2", "2M3", "2R1.5", "2R2.5",
    "3L1.5", "3L2.5", "3M1", "3M2", "3M3", "3R1.5", "3R2.5",
    "4L1.5", "4L2.5", "4M1", "4M2", "4M3", "4R1.5", "4R2.5",
    "5M1", "5M2", "5M3",
]


def normalize_shade_name(value: str) -> str:
    shade = str(value).strip().upper().replace(" ", "")
    replacements = {
        "2L15": "2L1.5",
        "2R15": "2R1.5",
        "3L15": "3L1.5",
        "3R15": "3R1.5",
        "4L15": "4L1.5",
        "4R15": "4R1.5",
    }
    return replacements.get(shade, shade)


def pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def extract_json_object(text: str) -> dict:
    if not text:
        raise ValueError("The model returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```json", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON object was found in the model response.")
    return json.loads(match.group(0))


def estimate_visual_shade(
    image: Image.Image,
    api_key: str,
    model: str = "gpt-5.6-luna",
    allowed_shades: list[str] | None = None,
    language: str = "English",
) -> dict:
    """Return an independent visual VITA 3D-Master estimate from a tooth ROI.

    This function deliberately receives only the image plus the allowed shade names.
    It does not receive calibrated CIELAB, CIEDE2000, Rayplicker, or other model results.
    """
    if OpenAI is None:
        raise RuntimeError("The OpenAI Python package is not installed.")

    shades = allowed_shades or ALLOWED_3D_MASTER_SHADES
    shades = [normalize_shade_name(s) for s in shades]
    allowed_text = ", ".join(shades)

    system_prompt = f"""
You are an independent multimodal comparison arm in a dental shade research application.

Examine only the supplied single-tooth image and make a visual estimate of the most likely
VITA 3D-Master shade. You are NOT given the calibrated CIELAB result, CIEDE2000 result,
Rayplicker reference result, or predictions from any other model.

Choose exactly one shade from this list:
{allowed_text}

Rules:
- Never invent a shade outside the list.
- This is a visual estimate, not a spectrophotometric measurement.
- Do not claim that you measured L*, a*, b* or DeltaE.
- Inspect image quality and flag glare, blur, exposure problems, and visible
  cervical/middle/incisal variation.
- confidence_percent is your model-reported confidence in the categorical visual estimate;
  it is not a calibrated probability.
- Return JSON only, with no markdown.

Required JSON keys:
predicted_shade
confidence_percent
image_quality
glare
blur
exposure
cervical_middle_incisal_variation
notes
""".strip()

    user_prompt = f"""
Estimate the most likely VITA 3D-Master shade for this tooth ROI.
Write descriptive fields in {language}. Return valid JSON only.
""".strip()

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": pil_to_data_url(image)},
                ],
            },
        ],
    )

    raw_text = response.output_text
    parsed = extract_json_object(raw_text)

    predicted = normalize_shade_name(parsed.get("predicted_shade", ""))
    if predicted not in shades:
        raise ValueError(
            f"Model returned invalid shade '{predicted}'. Allowed shades: {allowed_text}"
        )

    confidence = parsed.get("confidence_percent")
    try:
        confidence = int(round(float(confidence)))
        confidence = max(0, min(100, confidence))
    except Exception:
        confidence = None

    return {
        "predicted_shade": predicted,
        "confidence_percent": confidence,
        "image_quality": str(parsed.get("image_quality", "")),
        "glare": str(parsed.get("glare", "")),
        "blur": str(parsed.get("blur", "")),
        "exposure": str(parsed.get("exposure", "")),
        "cervical_middle_incisal_variation": str(
            parsed.get("cervical_middle_incisal_variation", "")
        ),
        "notes": str(parsed.get("notes", "")),
        "model": model,
        "raw_response": raw_text,
    }



def estimate_visual_lab(
    image: Image.Image,
    api_key: str,
    model: str = "gpt-5.6-luna",
    language: str = "English",
) -> dict:
    """Return a blinded zero-shot VLM estimate of apparent tooth CIELAB.

    The model receives only the tooth ROI. It is not given Rayplicker values,
    VITA labels/reference coordinates, calibrated CIELAB, or predictions from
    any other model. The output is an image-based estimate, not a physical
    colorimetric measurement.
    """
    if OpenAI is None:
        raise RuntimeError("The OpenAI Python package is not installed.")

    system_prompt = """
You are an independent multimodal research comparator in a dental color study.

Examine only the supplied single-tooth image and estimate the tooth's apparent
CIELAB coordinates. Do not assign a VITA shade and do not infer from any known
reference label. You are NOT given Rayplicker measurements, VITA shade labels,
VITA reference CIELAB values, calibrated software measurements, or predictions
from other models.

Estimate the representative tooth body color, prioritizing the middle third of
the visible crown and excluding obvious specular highlights, deep shadows,
gingiva, background, text, borders, and interface overlays.

Use conventional CIELAB notation:
- L_star: lightness, 0 to 100
- a_star: red-green axis
- b_star: yellow-blue axis

Important:
- These are visual model estimates from an image, NOT spectrophotometric
  measurements and NOT calibrated colorimetry.
- Do not force values to a named VITA shade.
- Do not output a VITA shade.
- confidence_percent is model-reported confidence in the visual CIELAB
  estimate and is not a calibrated probability.
- Return JSON only, with no markdown.

Required JSON keys:
L_star
a_star
b_star
confidence_percent
image_quality
glare
blur
exposure
notes
""".strip()

    user_prompt = f"""
Estimate the representative apparent CIELAB L*, a*, b* values of this tooth ROI.
Write descriptive fields in {language}. Return valid JSON only.
""".strip()

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": pil_to_data_url(image)},
                ],
            },
        ],
    )

    raw_text = response.output_text
    parsed = extract_json_object(raw_text)

    def _finite_float(name: str, lo: float, hi: float) -> float:
        value = float(parsed.get(name))
        if not (lo <= value <= hi):
            raise ValueError(
                f"Model returned {name}={value}, outside the prespecified range "
                f"[{lo}, {hi}]."
            )
        return value

    L_star = _finite_float("L_star", 0.0, 100.0)
    a_star = _finite_float("a_star", -128.0, 127.0)
    b_star = _finite_float("b_star", -128.0, 127.0)

    confidence = parsed.get("confidence_percent")
    try:
        confidence = int(round(float(confidence)))
        confidence = max(0, min(100, confidence))
    except Exception:
        confidence = None

    return {
        "pred_L": L_star,
        "pred_a": a_star,
        "pred_b": b_star,
        "confidence_percent": confidence,
        "image_quality": str(parsed.get("image_quality", "")),
        "glare": str(parsed.get("glare", "")),
        "blur": str(parsed.get("blur", "")),
        "exposure": str(parsed.get("exposure", "")),
        "notes": str(parsed.get("notes", "")),
        "model": model,
        "raw_response": raw_text,
    }
