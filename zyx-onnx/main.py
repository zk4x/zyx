import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.l0 = torch.nn.Linear(128, 8)

    def forward(self, x):
        return self.l0(x).relu()

input_tensor = torch.rand((8, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    #dynamo=True             # True or False to select the exporter to use
)
