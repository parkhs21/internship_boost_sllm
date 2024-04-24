import gradio as gr
from dataclasses import dataclass

@dataclass
class InputTab:
    tab: gr.Tab
    model_dd: gr.Dropdown
    rag_dd: gr.Dropdown
    ft_dd: gr.Dropdown
    token_slider: gr.Slider
    input_box: gr.Textbox
    output_box: gr.Textbox
    output_raw: gr.JSON
    submit_btn: gr.Button
    input_example: gr.Examples
    
    def __init__(self):
        with gr.Tab("데모") as self.tab:
            gr.Markdown("- 옵션을 선택하고 질의할 수 있습니다.")
            gr.Markdown("- RAG 및 FineTuning 옵션은 추후 업데이트 예정입니다.")
            gr.Markdown("- Max New Token 옵션은 토큰 수가 많아질 수록 많은 양을 생성합니다. 다만, 그에 따라 응답 시간이 증가합니다.")
            
            with gr.Row():
                self.model_dd = gr.Dropdown(
                    # model_list,
                    label="Model",
                    # value=model_list[0],
                    interactive=True
                    )
                
                self.rag_dd = gr.Dropdown(
                    label="RAG",
                    # value=rag_list[0],
                    interactive=True
                    )
                
                self.ft_dd = gr.Dropdown(
                    label="FineTuning",
                    # value=fine_tuning_list[0],
                    interactive=True
                    )
                
            self.token_slider = gr.Slider(
                minimum=96,
                maximum=1024,
                value=512,
                step=16,
                label="Max New Token",
                interactive=True
                )
            
            with gr.Row():
                self.input_box = gr.Text(label="Input", show_copy_button=True, lines=10)
                
            with gr.Accordion("Output_Raw", open=False):
                self.output_raw = gr.JSON(elem_id="output_raw", show_label=False)
                
            self.output_box = gr.Text(label="Ouptut", show_copy_button=True, lines=10)
            
            with gr.Row():
                gr.ClearButton([self.input_box, self.output_box, self.output_raw])
                self.submit_btn = gr.Button("Submit", variant="primary")

            # self.input_example = gr.Examples([])