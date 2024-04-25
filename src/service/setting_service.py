from ..network import MySession, MyResponse, MyException
import gradio as gr

class SettingService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session

    def reload_info(self):
        try:
            response: MyResponse = self.session.get_model_infos()
            
            gpu_info = {}
            for info in response.result["gpu_info"]:
                name = f'{info["name"]} ({info["memory.used [MiB]"]} / {info["memory.total [MiB]"]} MiB)'
                confidence = int(info["memory.used [MiB]"])/int(info["memory.total [MiB]"])
                gpu_info[name] = confidence
            
            return gpu_info, response.result["model_loaded_list"]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")


    def model_load(self, model: str, gpu: str):
        try:
            response: MyResponse = self.session.post_model_load(model, gpu)
            
            gpu_info = {}
            for info in response.result["gpu_info"]:
                name = f'{info["name"]} ({info["memory.used [MiB]"]} / {info["memory.total [MiB]"]} MiB)'
                confidence = int(info["memory.used [MiB]"])/int(info["memory.total [MiB]"])
                gpu_info[name] = confidence
            
            return gpu_info, response.result["model_loaded_list"]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")
        