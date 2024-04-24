import gradio as gr
from dataclasses import dataclass

@dataclass
class SettingTab:
    tab: gr.Tab
    gpu_usage_label: gr.Label
    model_loaded_hltext: gr.HighlightedText
    model_load_dd: gr.Dropdown
    gpu_load_dd: gr.Dropdown
    refresh_btn: gr.Button
    unload_btn: gr.Button
    load_btn: gr.Button
    
    def __init__(self):
        with gr.Tab("설정") as self.tab:
            gr.Markdown("- LLM 모델과 관련된 설정을 할 수 있습니다.")
            gr.Markdown("- 모델은 로드된 경우에만 사용 가능합니다.")
            
            self.gpu_usage_label = gr.Label(
                label="GPU 점유량",
                elem_id="gpu_graph"
            )
            
            self.model_loaded_hltext = gr.HighlightedText(
                # [["Nothing loaded\n", None]],
                label="Model Load 현황",
                combine_adjacent=True,
                adjacent_separator='',
                color_map={"GPU 0": "orange", "GPU 1": "orange", "GPU 2": "orange", "GPU 3": "orange"},
                elem_id="model_load_state"
            )
            
            self.refresh_btn = gr.Button("Refresh")
            
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
                self.unload_btn = gr.Button("Unload")
                self.load_btn = gr.Button("Load", variant="primary")