import warnings
warnings.filterwarnings("ignore")

import gradio as gr
from PIL import Image
import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# Device (Spaces CPU by default; enable GPU locally if available)
device = torch.device("cpu")
# For local GPU, replace the line above with:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once
image_classifier = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
).to(device).eval()
image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224"
)

caption_processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
caption_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device).eval()

def analyze_image(image):
    try:
        # Ensure PIL RGB
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Classification
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = image_classifier(**inputs).logits
            probs = logits.softmax(dim=1)

        top5_prob, top5_idx = torch.topk(probs, 5)
        classifications = {
            image_classifier.config.id2label[idx.item()]: float(prob.item())
            for idx, prob in zip(top5_idx[0], top5_prob[0])
        }

        # Caption
        cap_inputs = caption_processor(image, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = caption_model.generate(**cap_inputs, max_new_tokens=32)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)

        # Simple stats
        arr = np.array(image)
        avg = arr.mean(axis=(0, 1))
        brightness = float(arr.mean())
        w, h = image.size

        description = (
            f"Image Caption: {caption}\n\n"
            "Analysis:\n"
            f"- Brightness: {brightness:.1f}/255\n"
            f"- Average Color: R:{avg[0]:.1f}, G:{avg[1]:.1f}, B:{avg[2]:.1f}\n"
            f"- Resolution: {w}x{h}\n"
        )
        return classifications, description

    except Exception as e:
        return {"error": str(e)}, f"An error occurred: {e}"

CSS = """
body { background-color: white !important; color: black !important; }
h1 { font-size: 32px !important; font-weight: bold !important; text-align: center !important; color: black !important; }
label { font-size: 18px !important; color: black !important; }
.gradio-container { font-size: 16px !important; color: black !important; background-color: white !important; }
"""

with gr.Blocks(css=CSS) as demo:
    gr.Markdown("# Image Recognition System")
    gr.Markdown("Upload an image to analyze its content and get detailed information.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            output_labels = gr.Label(label="Classifications")
            output_text = gr.Textbox(label="Analysis Details", lines=10)

    submit_btn.click(analyze_image, inputs=[input_image], outputs=[output_labels, output_text])

if __name__ == "__main__":
    demo.launch(share=True)