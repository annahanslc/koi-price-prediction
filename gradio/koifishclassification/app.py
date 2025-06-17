import gradio as gr
from predict import predict_image


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs=["text"],
)

demo.launch()
