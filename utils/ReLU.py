class ReLu:
    def __init__(self,name):
        self.name=name
    def forward(self, inputs):
        self.inputs = inputs
        ret = inputs.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, dy, lr):
        dx = dy*(self.inputs >= 0)
        return dx
    def extract(self):
        return