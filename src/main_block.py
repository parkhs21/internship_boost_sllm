import gradio as gr
from .tab import SettingTab, InputTab

class MainBlock:
    block: gr.Blocks = gr.Blocks()
    setting_tab: SettingTab
    input_tab: InputTab

    def __init__(self):
        with gr.Blocks() as self.block:
            gr.Markdown("# Boost LLM")
            with gr.Row():
                with gr.Column():
                    self.setting_tab = SettingTab()
                with gr.Column(scale=2):
                    self.input_tab = InputTab()