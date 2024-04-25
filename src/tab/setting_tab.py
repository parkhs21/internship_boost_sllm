from ..network import MySession
from ..service import SettingService
import gradio as gr
from pathlib import Path

class SettingTab:
    service: SettingService
    tab: gr.Tab
    gpu_usage_label: gr.Label
    model_loaded_hltext: gr.HighlightedText
    model_load_dd: gr.Dropdown
    gpu_load_dd: gr.Dropdown
    refresh_btn: gr.Button
    # unload_btn: gr.Button
    load_btn: gr.Button
    
    def __init__(self, session: MySession):
        self.service = SettingService(session)
        with gr.Tab("설정") as self.tab:
            with gr.Blocks() as block:
                gr.Markdown("- LLM 모델과 관련된 설정을 할 수 있습니다.")
                gr.Markdown("- 모델은 로드된 경우에만 사용 가능합니다.")
                gr.Markdown("- 30분 동안 사용되지 않은 모델은 자동으로 언로드 됩니다.")
                gr.Markdown("- Model 명칭은 [Hugging Face]에서 추출하였습니다.")
                
                self.gpu_usage_label = gr.Label(
                    label="GPU 점유량",
                    elem_id="gpu_graph"
                )
                
                self.model_loaded_hltext = gr.HighlightedText(
                    # [["Nothing loaded\n", None]],
                    label="Model Load 현황",
                    combine_adjacent=True,
                    adjacent_separator='',
                    color_map={"GPU A6000": "orange", "GPU A5500": "orange", "GPU A5000": "orange", "GPU A4000": "orange"},
                    elem_id="model_load_state"
                )
                
                with gr.Row():
                    self.model_load_dd = gr.Dropdown(
                        # model_list,
                        label="Model",
                        # value=model_list[0],
                        interactive=True,
                        scale=4
                    )
                    
                    self.gpu_load_dd = gr.Dropdown(
                        # gpu_list,
                        label="GPU",
                        # value=gpu_list[1],
                        interactive=True
                    )
                    
                with gr.Row():
                    # self.unload_btn = gr.Button("Unload")
                    self.refresh_btn = gr.Button("Refresh")
                    self.load_btn = gr.Button("Load", variant="primary")


            block.load(
                fn=self.service.initial_render,
                outputs=[
                    self.gpu_usage_label,
                    self.model_loaded_hltext,
                    self.model_load_dd,
                    self.gpu_load_dd
                ]
            ).then(fn=None, js=Path("./src/js/setting_initial_render.js").read_text())

            self.refresh_btn.click(
                fn=self.service.reload_info,
                outputs=[
                    self.gpu_usage_label,
                    self.model_loaded_hltext
                ]
            )
            
            self.load_btn.click(
                fn=self.service.model_load,
                inputs=[
                    self.model_load_dd,
                    self.gpu_load_dd
                ],
                outputs=[
                    self.gpu_usage_label,
                    self.model_loaded_hltext
                ]
            )