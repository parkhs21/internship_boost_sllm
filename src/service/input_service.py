from ..network import MySession, MyResponse, MyException
import gradio as gr

class InputService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session

    def initial_render(self):
        try:
            model_res: MyResponse = self.session.get_model_cached()
            rag_res = ["Not Ready"]
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
        
    def completion(self, model: str, rag: str, ft: str, token: int, input: str):
        res = self.session.post_gen_text(model, rag, ft, token, input)
        output = res.result["output_text"]
        
        res.result["input_text"] = "Input (질의텍스트) 참조"
        res.result["output_text"] = "Output (답변텍스트) 참조"
        
        return output, res.result, gr.update(value=res.result["time"])
        