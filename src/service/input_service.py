from ..network import MySession

class InputService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session