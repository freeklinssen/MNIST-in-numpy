

class sequential():
    # layers sould be a list
    def __init__(self, layers):
        self.training = True
        
        self.Layers = []
        for layer in layers:
            self.Layers.append(layer)
            
    
    def forward(self, x):
        for layer in self.Layers:
              x = layer.forward(x)
        return x
    
    def backward(self, error):
        for layer in self.Layers[::-1]:
              error = layer.backward(error)
        return error

    def Train(self):
        for layer in self.Layers:
            if hasattr(layer, 'Train'):
                layer.Train()
    
    def Eval(self):
        for layer in self.Layers:
            if hasattr(layer, 'Eval'):
                layer.Eval()
            
    def Optimizer(self, optimizer):
        for layer in self.Layers:
            if hasattr(layer, 'Optimizer'):
                layer.Optimizer(optimizer)


    