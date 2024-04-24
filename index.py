import gradio as gr
from src.tab import InputTab, SettingTab

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            st = SettingTab()
        with gr.Column(scale=2):
            it = InputTab()
            
if __name__=='__main__':
    demo.launch(server_name="0.0.0.0", server_port=7863)