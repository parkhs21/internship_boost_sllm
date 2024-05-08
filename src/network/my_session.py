from .my_response import MyResponse
from .my_exception import MyException
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class MySession():
    """
    API 통신을 위한 클래스입니다.
    ---------------------------
    """
    _session: requests.Session
    _endpoint: str
    
    def __init__(self):
        self._session = requests.Session()
        self._session.auth = (os.environ.get('username'), os.environ.get('password'))
        self._endpoint = os.environ.get('endpoint')
        
    def get(self, detail: str, params: dict = None) -> requests.models.Response:
        response = self._session.get(self._endpoint + detail, params=params)
        response = MyResponse(response)
        if not response.isSuccess:
            raise MyException(response.result)
        return response
    
    def post(self, detail: str, body: dict = None) -> requests.models.Response:
        response = self._session.post(self._endpoint + detail, json=body)
        response = MyResponse(response)
        if not response.isSuccess:
            raise MyException(response.result)
        return response

    def get_model_cached(self) -> MyResponse:
        return self.get("/model/list")
    
    def get_model_loaded(self) -> MyResponse:
        return self.get("/model/loaded-list")
    
    def get_gpu_info(self) -> MyResponse:
        return self.get("/gpu-info")
    
    def get_model_infos(self) -> MyResponse:
        return self.get("/model/loaded")

    def post_model_load(self, model: str, gpu: str) -> MyResponse:
        body = {
            "model": model,
            "gpu_index": gpu
        }
        return self.post("/model/load", body)
    
    def post_gen_text(self, model: str, rag: str, ft: str, token: int, input: str) -> MyResponse:
        body = {
            "prompt": input,
            "model": model,
            "max_new_token": token
        }
        return self.post("/generate", body)
    
    def post_gen_stream(self, model: str, rag: str, ft: str, sample: bool, temp: float, topp: float, token: int, input: str):
        body = {
            "prompt": input,
            "model": model,
            "max_new_token": token,
            "do_sample": sample,
            "temperature": temp,
            "top_p": topp
        }
        response = self._session.post(self._endpoint + "/generate/stream", json=body, stream=True)
        for content in response.iter_content():
            decoded_content = content.decode('utf-8')
            yield decoded_content
