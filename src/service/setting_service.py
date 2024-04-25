from ..network import MySession

class SettingService:
    session: MySession
    
    def __init__(self, session: MySession):
        self.session = session
    