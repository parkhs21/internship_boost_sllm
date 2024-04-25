from ..network import MySession, MyResponse, MyException
import gradio as gr

class SettingService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session
        
    def initial_render(self):
        try:
            model_infos_res: MyResponse = self.session.get_model_infos()
            model_cached_res: MyResponse = self.session.get_model_cached()
            gpu_list_res = [
                "A6000",
                "A5500",
                "A5000",
                "A4000"
            ]
            
            gpu_info = {}
            for info in model_infos_res.result["gpu_info"]:
                name = f'{info["name"]} ({info["memory.used [MiB]"]} / {info["memory.total [MiB]"]} MiB)'
                confidence = int(info["memory.used [MiB]"])/int(info["memory.total [MiB]"])
                gpu_info[name] = confidence
            
            return [
                gpu_info,
                model_infos_res.result["model_loaded_list"],
                gr.update(choices=model_cached_res.result, value=model_cached_res.result[0]),
                gr.update(choices=gpu_list_res, value=gpu_list_res[0]),
            ]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")

    def reload_info(self):
        try:
            res: MyResponse = self.session.get_model_infos()
            
            gpu_info = {}
            for info in res.result["gpu_info"]:
                name = f'{info["name"]} ({info["memory.used [MiB]"]} / {info["memory.total [MiB]"]} MiB)'
                confidence = int(info["memory.used [MiB]"])/int(info["memory.total [MiB]"])
                gpu_info[name] = confidence
            
            return gpu_info, res.result["model_loaded_list"]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")


    def model_load(self, model: str, gpu: str):
        try:
            res: MyResponse = self.session.post_model_load(model, gpu)
            
            gpu_info = {}
            for info in res.result["gpu_info"]:
                name = f'{info["name"]} ({info["memory.used [MiB]"]} / {info["memory.total [MiB]"]} MiB)'
                confidence = int(info["memory.used [MiB]"])/int(info["memory.total [MiB]"])
                gpu_info[name] = confidence
            
            return gpu_info, res.result["model_loaded_list"]
        except MyException as e:
            raise gr.Error(e.msg)
        except:
            raise gr.Error("네트워크 통신에 실패했습니다.")
        