# Copyright 2020 Pascal Philipp

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from keras.layers import Layer
from keras.initializers import RandomNormal
import tensorflow as tf


class ReflectionLayer(Layer):

    def __init__(self, **kwargs):
        self.reflector = None
        super(ReflectionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        rn_args = {'mean': 0, 'stddev': 0.1}
        self.reflector = self.add_weight(name='kernel',
                                         shape=input_shape[1:],
                                         initializer=RandomNormal(**rn_args),
                                         trainable=True)
        super(ReflectionLayer, self).build(input_shape)

    def call(self, x):
        scalars = tf.reduce_sum(tf.multiply(x, self.reflector),
                                axis=tuple(range(1, len(x.shape))),
                                keepdims=True)
        prod = tf.multiply(tf.expand_dims(self.reflector, 0), scalars)
        return (2 / tf.reduce_sum(tf.multiply(self.reflector,
                                              self.reflector))) * prod - x

    def compute_output_shape(self, input_shape):
        return input_shape
