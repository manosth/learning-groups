import numpy as np
class Params():
        def __init__(self):
            # optimizer and training params
            self.lr = 1e-2
            self.eps = 1e-8

            self.batch_size = 128
            self.epochs = 300

            self.tensorboard = True
            self.dataset = "cifar"

            # model params
            self.model = "suul" # sparse, untied, whitened, learned bias
            self.group_size = 4
            self.num_groups = 5
            self.kernel_size = 6
            self.stride = 1

            self.num_layers = 4

            self.step = 0.01
            self.lmbd = 0

            self.lam_loss = 0

            # data params
            self.n_classes = 10
            self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

            self.n_channels = 3
            self.input_size = 1024
            self.input_width = 32
            self.input_height = 32

            # classification params
            self.pool_size = 4
