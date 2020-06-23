class reasoningSystem():
    def __init__(self):
        pass
    def init(self, info, training):
        self.info = info
        self.training = training
        self.model = None

    def update(self, frame, state):
        if self.training :
            pass
        solution = self.get(frame, state)
        return solution

    def get(self, frame, state):
        solution = None # self.model(frame, state)
        return solution
reasoning : reasoningSystem = reasoningSystem()
