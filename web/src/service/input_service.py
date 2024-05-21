from ..network import MySession, MyResponse, MyException
import gradio as gr
import time

class InputService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session

    def initial_render(self):
        try:
            model_res: MyResponse = self.session.get_model_cached()
            rag_res = ["Not Ready", "대한민국 민법"]
            ft_res = ["Not Ready"]
            
            return [
                gr.update(choices=model_res.result, value=model_res.result[0]),
                gr.update(choices=rag_res, value=rag_res[0]),
                gr.update(choices=ft_res, value=ft_res[0]),
                None
            ]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")
        
    def sample_check_toggle(self, isChecked: bool):
        if isChecked:
            return gr.update(interactive=True), gr.update(interactive=True)
        else:
            return gr.update(value=1, interactive=False), gr.update(value=1, interactive=False)
        
    def clear_text(self):
        return gr.update(value=None), gr.update(value=None), gr.update(value=None)
        
    def completion(self, model: str, rag: str, ft: str, token: int, input: str):
        res = self.session.post_gen_text(model, rag, ft, token, input)
        output = res.result["output_text"]
        
        res.result["input_text"] = "Input (질의텍스트) 참조"
        res.result["output_text"] = "Output (답변텍스트) 참조"
        
        return output, res.result, gr.update(value=res.result["time"])
    
    def completion_stream(self, model: str, rag: str, ft: str, sample: bool, temp: float, topp: float, token: int, input: str):
        output = ""
        if rag=="Not Ready":
            rag = None
        response = self.session.post_gen_stream(model, rag, ft, sample, temp, topp, token, input)
        for res in response:
            output += res
            yield output
         