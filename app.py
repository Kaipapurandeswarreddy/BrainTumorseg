import gradio as gr
import os
import matplotlib.pyplot as plt
import numpy as np
from inference import run_inference
from google import genai
from google.genai import types

# -----------------------------
# AI EXPLANATION FUNCTION
# -----------------------------
def generate_explanation(volumes, api_key):
    """Generates a medical explanation using the new Google GenAI SDK (User Pattern)."""
    if not api_key:
        return "⚠️ API Key missing. Please provide a valid Google Gemini API key."
    
    # Try models in order: Flash 2.0 (User requested), 1.5 Flash
    models_to_try = [
        "gemini-2.5-flash", 
    ]

    try:
        # New SDK Initialization
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are a medical AI assistant. Analyze the following brain tumor data derived from a BraTS segmentation:
        
        {volumes}
        
        Provide a concise, professional summary for a radiologist or neurologist. 
        Explain what these volumes indicate about the tumor burden.
        Do not provide a diagnosis, but suggest potential clinical implications based on standard medical knowledge.
        Keep it under 150 words.
        """
        
        last_error = ""
        for model_name in models_to_try:
            try:
                # User's provided pattern: contents=[types.Part.from_text(...)], config=...
                response = client.models.generate_content(
                    model=model_name,
                    contents=[types.Part.from_text(text=prompt)],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        top_p=0.95,
                        top_k=20,
                    ),
                )
                return f"🤖 Analysis by {model_name}:\n\n" + response.text
            except Exception as e:
                last_error = str(e)
                continue
        
        return f"❌ All Gemini models failed. Last error: {last_error}"

    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}"

# -----------------------------
# PLOT FUNCTION
# -----------------------------
def plot_slice(flair_volume, seg_volume, slice_idx):
    """Refactored plotting logic to return an image array."""
    if flair_volume is None or seg_volume is None:
        return None
    
    # Ensure slice_idx is valid
    D = flair_volume.shape[2]
    slice_idx = min(max(0, slice_idx), D - 1)

    fig = plt.figure(figsize=(6,6))
    plt.imshow(flair_volume[:, :, slice_idx], cmap="gray")
    plt.imshow(seg_volume[:, :, slice_idx], cmap="tab10", alpha=0.5)
    plt.axis("off")
    
    # Convert figure to numpy array
    fig.canvas.draw()
    # Handle both old and new matplotlib versions or just use buffer_rgba()
    try:
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # For matplotlib 3.8+
        image_from_plot = np.asarray(fig.canvas.buffer_rgba())
    
    plt.close(fig)
    return image_from_plot

# -----------------------------
# GRADIO CALLBACKS
# -----------------------------
def predict(t1_file, t1ce_file, t2_file, flair_file, ckpt_file, api_key):
    if not all([t1_file, t1ce_file, t2_file, flair_file, ckpt_file]):
        return None, "Please upload all files.", None, None, None, gr.Slider(value=0, maximum=0), ""

    # Gradio passes file paths as strings (temp files)
    t1_path = t1_file.name
    t1ce_path = t1ce_file.name
    t2_path = t2_file.name
    flair_path = flair_file.name
    ckpt_path = ckpt_file.name

    try:
        # Run inference and get 3D volumes + best slice index
        output_path, volumes, flair_np, seg_np, best_z = run_inference(
            t1_path, t1ce_path, t2_path, flair_path, ckpt_path
        )
        
        # Initial plot
        img = plot_slice(flair_np, seg_np, best_z)
        
        # Generate AI Explanation
        explanation = generate_explanation(volumes, api_key)
        
        return (
            output_path, 
            volumes, 
            img, 
            flair_np, 
            seg_np, 
            gr.Slider(value=int(best_z), maximum=int(flair_np.shape[2]-1), interactive=True),
            explanation
        )
        
    except Exception as e:
        return None, {"error": str(e)}, None, None, None, gr.Slider(value=0, maximum=1, interactive=False), f"Error: {str(e)}"

def update_view(slice_val, flair_vol, seg_vol):
    """Callback to update plot when slider moves."""
    return plot_slice(flair_vol, seg_vol, int(slice_val))

# -----------------------------
# APP LAYOUT & THEME
# -----------------------------
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"]
)

css = """
.container { max-width: 1200px; margin: auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
.header p { font-size: 1.1rem; color: #64748b; }
.run-btn { margin-top: 1rem; }
"""

with gr.Blocks(theme=theme, css=css, title="BraTS Tumor Segmentation") as app:
    with gr.Column(elem_classes=["container"]):
        with gr.Column(elem_classes=["header"]):
            gr.Markdown("# 🧠 BraTS 21: Brain Tumor Segmentation")
            gr.Markdown("State-of-the-art 3D volumetric segmentation using MONAI and patches-based SegResNet.")

        # State variables
        flair_state = gr.State()
        seg_state = gr.State()

        with gr.Row(equal_height=False):
            # Left Column: Inputs
            with gr.Column(variant="panel", scale=1):
                gr.Markdown("### 📂 Data Upload")
                gr.Markdown("Upload the 4 MRI modalities (NIfTI format) and the trained model checkpoint.")
                
                with gr.Group():
                    with gr.Row():
                        t1_input = gr.File(label="T1-weighted", file_count="single", file_types=[".nii.gz", ".nii", ".gz"], height=100)
                        t1ce_input = gr.File(label="T1-Contrast", file_count="single", file_types=[".nii.gz", ".nii", ".gz"], height=100)
                    with gr.Row():
                        t2_input = gr.File(label="T2-weighted", file_count="single", file_types=[".nii.gz", ".nii", ".gz"], height=100)
                        flair_input = gr.File(label="FLAIR", file_count="single", file_types=[".nii.gz", ".nii", ".gz"], height=100)
                
                ckpt_input = gr.File(label="Model Checkpoint (.pt)", file_count="single", file_types=[".pt", ".pth", ".ckpt"], height=100)
                
                gr.Markdown("### 🤖 AI Settings")
                api_key_input = gr.Textbox(
                    label="Google Gemini API Key", 
                    value="",
                    type="password",
                    placeholder="AIzaSy..."
                )

                run_btn = gr.Button("🚀 Run Segmentation & Analysis", variant="primary", size="lg", elem_classes=["run-btn"])

            # Right Column: Results
            with gr.Column(variant="panel", scale=2):
                gr.Markdown("### 🔍 Visualization & Analysis")
                
                # Plot
                plot_output = gr.Image(
                    label="Tumor Segmentation Overlay", 
                    type="numpy", 
                    interactive=False, 
                    height=400,
                    show_download_button=False
                )
                
                # Slider
                slice_slider = gr.Slider(
                    minimum=0, maximum=154, value=75, step=1, 
                    label="Axial Slice Navigation", 
                    info="Slide to explore the 3D volume",
                    interactive=True
                )
                
                # AI Explanation Box
                explanation_output = gr.Textbox(label="🤖 AI Medical Analysis (Gemini Flash)", lines=4, interactive=False)
                
                with gr.Row():
                    volume_output = gr.JSON(label="Calculated Tumor Volumes (cm³)")
                    file_output = gr.File(label="Download 3D Segmentation", height=100)

    # -----------------------------
    # EVENT HANDLERS
    # -----------------------------
    run_btn.click(
        fn=predict,
        inputs=[t1_input, t1ce_input, t2_input, flair_input, ckpt_input, api_key_input],
        outputs=[file_output, volume_output, plot_output, flair_state, seg_state, slice_slider, explanation_output]
    )
    
    slice_slider.change(
        fn=update_view,
        inputs=[slice_slider, flair_state, seg_state],
        outputs=[plot_output],
        show_progress=False
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
