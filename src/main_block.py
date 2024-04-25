import gradio as gr
from .tab import SettingTab, InputTab
from .network import MySession

class MainBlock:
    block: gr.Blocks
    setting_tab: SettingTab
    input_tab: InputTab
    session: MySession

    def __init__(self):
        self.session = MySession()
        with gr.Blocks() as self.block:
            gr.Markdown("# Boost LLM")
            with gr.Row():
                with gr.Column():
                    self.setting_tab = SettingTab(self.session)
                with gr.Column(scale=2):
                    self.input_tab = InputTab(self.session)