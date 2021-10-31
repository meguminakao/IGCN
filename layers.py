from __future__ import division
import tensorflow.compat.v1 as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}

def uniform(shape, scale=0.05, name=None):
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def project(img_feat, x, y, dim):

    x1 = tf.floor(x)
    x2 = tf.ceil(x)
    y1 = tf.floor(y)
    y2 = tf.ceil(y)

    Q11 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y1,tf.int32)],1))
    Q12 = tf.gather_nd(img_feat, tf.stack([tf.cast(x1,tf.int32), tf.cast(y2,tf.int32)],1))
    Q21 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y1,tf.int32)],1))
    Q22 = tf.gather_nd(img_feat, tf.stack([tf.cast(x2,tf.int32), tf.cast(y2,tf.int32)],1))

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
    Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y2,y))
    Q21 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q21)

    weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y,y1))
    Q12 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q12)

    weights = tf.multiply(tf.subtract(x,x1), tf.subtract(y,y1))
    Q22 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q22)

    outputs = tf.add_n([Q11, Q21, Q12, Q22])
    return outputs

def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False


    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    # Graph convolution layer

    def __init__(self, input_dim, output_dim, placeholders, dropout=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, gcn_block_id = 1,
                 featureless=False, **kwargs):

        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
             
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GraphPooling(Layer):
	# Graph unPooling layer
	def __init__(self, placeholders, pool_id=1, **kwargs):
		super(GraphPooling, self).__init__(**kwargs)

		self.pool_idx = placeholders['pool_idx'][pool_id-1]

	def _call(self, inputs):
		X = inputs

		add_feat = (1/2.0) * tf.reduce_sum(tf.gather(X, self.pool_idx), 1)
		outputs = tf.concat([X, add_feat], 0)

		return outputs
        
class GraphProjection(Layer):
    
    # graph projection layer 
    def __init__(self, placeholders, **kwargs):
        super(GraphProjection, self).__init__(**kwargs)

        self.img_feat = placeholders['img_feat']
        self.shapes = placeholders['shapes']
        self.ipos = placeholders['ipos']
      
    def _call(self, inputs): 

        # inputs, shape = (500, 3)
        coord = inputs

        # 1.0 in RGB space = 6800 / rmax mm in XYZ real space --> render.py
        co = 1.0 / 6800.0
        shift = 122.0
            
        x = self.ipos[:, 1]
        y = self.ipos[:, 0]
        d = (project(self.img_feat[0], y, x, 3) - shift ) * co

        outputs = tf.concat([d, self.shapes], 1)

        return outputs
