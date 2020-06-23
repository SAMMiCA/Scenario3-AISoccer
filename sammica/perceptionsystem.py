class perceptionSystem():
    def __init__(self):
        pass
    def init(self, info, training):
        self.info = info
        self.training = training
        self.model = None

    def update(self, frame):
        if self.training :
            pass
        state = self.get(frame)
        return state

    def get(self, frame):
        state = None # self.model(frame)
        return state