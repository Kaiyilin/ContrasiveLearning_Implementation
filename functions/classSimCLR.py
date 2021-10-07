import tensorflow as tf
from decimal import Decimal
import numpy as np
 
class SimCLR(tf.keras.Model):

    def __init__(self, backboneNet, projection_head) -> None:
        """ The initial SimCLR should have a backbone model"""
        super(SimCLR, self).__init__()
        self.Net = backboneNet
        self.projection_head = projection_head

    def compile(self, optimiser, loss_fn):
        super(SimCLR, self).compile()
        self.opt = optimiser
        self.loss = loss_fn

    def __str__():
        print("SimCLR is working")

    def train_step(self, data):
        """ The input of for this should have two images
        
        """
        img1 = data[0]
        img2 = data[1]

        with tf.GradientTape(persistent=True) as tape:
            representation1 = self.Net(img1)
            representation2 = self.Net(img2)
            representation1 = tf.math.l2_normalize(representation1, axis=1)
            representation2 = tf.math.l2_normalize(representation2, axis=1)

            similarityLoss = self.loss(representation1, representation2)

            Net_gradients = tape.gradient(similarityLoss, 
                                            self.Net.trainable_variables)
            

            
            # Apply the gradients to the optimizer
            self.opt.apply_gradients(zip(Net_gradients, 
                                        self.Net.trainable_variables))


        return {"Similarity_Loss": similarityLoss}
            
