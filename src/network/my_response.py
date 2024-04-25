from dataclasses import dataclass
from typing import Union
import requests

@dataclass
class MyResponse:
    """
    API 통신 반환 DTO입니다.
    ----------------------
    
    isSuccess: bool
    result: str
    """
    
    isSuccess: bool
    result: str

    def __init__(self, response: Union[requests.models.Response, dict]):
        
        if isinstance(response, requests.models.Response):
            response = response.json()

        self.isSuccess = response["isSuccess"]
        self.result = response["result"]
