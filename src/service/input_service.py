from ..network import MySession, MyResponse, MyException
import gradio as gr

class InputService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session

    def initial_render(self):
        try:
            model_cached_res: MyResponse = self.session.get_model_cached()
            return [
                gr.update(choices=model_cached_res.result, value=model_cached_res.result[0]),
                None
            ]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")