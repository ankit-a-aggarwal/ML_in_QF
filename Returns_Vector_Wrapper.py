
import tensorflow as tf
#from main_model import MainEncoderDecoder
from tensorflow.keras import layers, models
import numpy as np
class Returns_Vector_Linear_Wrapper(models.Model):
    def __init__(self,**kwargs):

        super(Returns_Vector_Linear_Wrapper, self).__init__(name="Returns Vector Linear Projection", **kwargs)

    def build(self, returns_vector):

        self.w = self.add_weight(name="Estimated_Returns_Trainable_Weight_Vector",
                                 shape=(returns_vector[0], returns_vector[1]),
                                 initializer='ones',
                                 constraint= tf.keras.constraints.non_neg(),#tf.keras.constraints.unit_norm(axis = 0)
                                 #constraint = tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0),
                                 trainable=True,
                                 )
        """
        self.b = self.add_weight(name='b',
                                 shape=(returns_vector.shape[1],),
                                 initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04), trainable=True
                                 )
        """


    def call(self, inputs, scope=None):

        #weights = self.w
        #self.w = self.w/tf.math.reduce_sum(self.w)
        sum = tf.math.reduce_sum(self.w)
        answer  = tf.math.divide(self.w, sum)
        self.w.assign(answer)
        #assert (tf.math.reduce_sum(self.w) == 1)
        indices = np.argsort(np.concatenate(self.w.numpy(),axis=0))
        #print(indices[:200])
        output = tf.matmul(tf.transpose(inputs,perm = [1,0]),self.w)#w [100*1]

        #output = output + self.b

        return output