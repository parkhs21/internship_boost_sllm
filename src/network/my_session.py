from .my_response import MyResponse
import requests
import os

class MySession():
    """
    API 통신을 위한 클래스입니다.
    ---------------------------
    """
    _session: requests.Session
    _endpoint: str
    
    def __init__(self):
        self._session = requests.Session()
        self._session.auth = (os.environ['username'], os.environ['password'])
        self._endpoint = os.environ['endpoint']
        
    def get(self, detail: str, params: dict = None) -> requests.models.Response:
        response = self._session.get(self._endpoint + detail, params=params)
        return response
    
    def post(self, detail: str, body: dict = None) -> requests.models.Response:
        response = self._session.post(self._endpoint + detail, json=body)
        return response

    def get_model_cached(self) -> MyResponse:
        response = self.get("/model/list")
        return MyResponse(response)
    
    def get_model_loaded(self) -> MyResponse:
        response = self.get("/model/loaded-list")
        return MyResponse(response)
    
    def get_gpu_info(self) -> MyResponse:
        response = self.get("/gpu-info")
        return MyResponse(response)
    
    # def post_model_load(self, model_id: str, gpu_id: str) -> MyResponse:
    #     body = {
    #         "mode_id": model_id,
    #         "gpu_id": gpu_id
    #     }
    #     response = self.post("/model/load", body)
    #     return MyResponse(response)
    
    # def post_gen_text(self, model_id: str, input: str) -> MyResponse:
    #     body = {
    #         "mode_id": model_id,
    #         "input": input
    #     }
    #     response = self.post("/generate", body)
    #     return MyResponse(response)

    def post_model_load(self, temp: int) -> MyResponse:
        body = {
            "temp": temp
        }
        response = self.post("/model/load", body)
        return MyResponse(response)
    
    def post_gen_text(self, temp: int) -> MyResponse:
        body = {
            "temp": temp
        }
        response = self.post("/generate", body)
        print(response.content)
        return MyResponse(response)