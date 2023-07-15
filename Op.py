class Op(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        return NotImplementedError
    
    def backward(self, grad_outputs):
        return NotImplementedError