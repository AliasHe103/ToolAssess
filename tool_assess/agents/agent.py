class Agent:
    def __init__(self, name):
        self.client = None
        self.name = name

    def predict(self, messages):
        raise NotImplementedError("Please override this method!")

    def test(self):
        raise NotImplementedError("Please override this method!")