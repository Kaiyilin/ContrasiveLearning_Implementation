from functions.All_functions import *
from functions.cm_tensorboard import *
from resnet3d import Resnet3DBuilder 
from ContrasiveLearningData import *
import logging

import math
from decimal import Decimal
import numpy as np
 
class SimCLR(tf.keras.Model):

    def __init__(self, Net1) -> None:
        """The Network architecture and weights should be the same
        """
        super(SimCLR, self).__init__()
        self.Net1 = Net1



    
    def compile(self, optNet1, loss_fn):
        super(SimCLR, self).compile()
        self.optNet1 = optNet1
        self.loss = loss_fn

    def __str__():
        print("SimCLR is working")

    def train_step(self, data):
        img1 = data[0]
        img2 = data[1]

        with tf.GradientTape(persistent=True) as tape:
            representation1 = self.Net1(img1)
            representation2 = self.Net1(img2)
            representation1 = tf.math.l2_normalize(representation1, axis=1)
            representation2 = tf.math.l2_normalize(representation2, axis=1)

            similarityLoss = self.loss(representation1, representation2)

            Net1_gradients = tape.gradient(similarityLoss, 
                                            self.Net1.trainable_variables)
            

            
            # Apply the gradients to the optimizer
            self.optNet1.apply_gradients(zip(Net1_gradients, 
                                            self.Net1.trainable_variables))


        return {"Similarity_Loss": similarityLoss}
            

logging.basicConfig(filename = 'Execution.log', level = logging.WARNING, format = '%(filename)s %(message)s')
execu_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model = Resnet3DBuilder.build_resnet_50((64, 64, 64, 1), 3,reg_factor=1e-4)
resCL1 = base_model_creator(model)

ResSimCLR = SimCLR(resCL1)
del model, resCL1



ResSimCLR.compile(optNet1 = opt3,
                  loss_fn= tf.keras.losses.CosineSimilarity())
ResSimCLR.fit(ds_tr.shuffle(50).map(tf_random_rotate_image_xyz).batch(50),
               epochs = 5)


