
import tensorflow as tf
#from main_model import MainEncoderDecoder
from tensorflow.keras import layers, models


class Covariance_Matrix_Linear_Wrapper(models.Model):

    def __init__(self,**kwargs):

        super(Covariance_Matrix_Linear_Wrapper, self).__init__(trainable=True,name="Covariance Matrix Linear Projection",dynamic=True, **kwargs)

    def build(self, returns_vector):

        self.w = self.add_weight(name="Estimated_Covariance_Matrix_Weight_Vector",
                                 shape=(returns_vector[0], returns_vector[1]),
                                 initializer='ones',
                                 constraint= tf.keras.constraints.non_neg(),#tf.keras.constraints.unit_norm(axis = 0)
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
        sum = tf.math.reduce_sum(self.w)
        answer = tf.math.divide(self.w, sum)
        self.w = answer
        #self.w.assign(answer)
        #assert (tf.math.reduce_sum(self.w) == 1)

        output = tf.matmul(tf.transpose(inputs,perm = [1,0]),self.w)

        #output = output + self.b

        return output
