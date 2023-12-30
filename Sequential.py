import os

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
    
    def Save_model(self, model_name):
        if os.path.exists(model_name):
            for file_name in os.listdir(model_name):
                file_path = os.path.join(model_name, file_name)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            
        for layer_id, layer in enumerate(self.Layers):
            if hasattr(layer, 'Save_layer'):
                print('okay')
                layer.Save_layer(model_name, layer_id)
    
    def Load_model(self, model_name):
        for layer_id, layer in enumerate(self.Layers):
            if hasattr(layer, 'Load_layer'):
                layer.Load_layer(model_name, layer_id)


    