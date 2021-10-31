from __future__ import division
import tflearn
from tflearn import conv_2d, merge
from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS
path = "Output/"

def mesh_loss(pred, placeholders):

    # Positional loss
    coord_init = placeholders['features']
    coord_pred = coord_init + pred
    coord_label = placeholders['labels']

    V_max = tf.reduce_max(coord_label)
    V_min = tf.reduce_min(coord_label)
    coord_pred = tf.divide(tf.subtract(coord_pred,V_min),tf.subtract(V_max,V_min))
    coord_label = tf.divide(tf.subtract(coord_label,V_min),tf.subtract(V_max,V_min))

    # Laplacian loss
    rowsum = tf.reduce_sum(placeholders['adj'],1)
    row = tf.stack([rowsum,rowsum,rowsum],1) 
    laplacian_pred = tf.subtract(coord_pred,tf.divide(tf.matmul(placeholders['adj'], coord_pred),row))
    laplacian_init = tf.subtract(coord_init,tf.divide(tf.matmul(placeholders['adj'], coord_init),row))
    laplacian_label = tf.subtract(coord_label,tf.divide(tf.matmul(placeholders['adj'], coord_label),row))

    L_max = tf.reduce_max(laplacian_label)
    L_min = tf.reduce_min(laplacian_label)
    laplacian_pred = tf.divide(tf.subtract(laplacian_pred,L_min),tf.subtract(L_max,L_min))
    laplacian_init = tf.divide(tf.subtract(laplacian_init,L_min),tf.subtract(L_max,L_min))
    laplacian_label = tf.divide(tf.subtract(laplacian_label,L_min),tf.subtract(L_max,L_min))

    positional_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(coord_pred, coord_label)),1))
    laplacian_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(laplacian_pred, laplacian_label)), 1))

    total_loss = positional_loss + 0.1 * laplacian_loss

    return total_loss


class Model:
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.loss = 0

        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        self.activations.append(self.inputs)
        
        eltwise = [3,5,7,9]
        for idx,layer in enumerate(self.layers):
            hidden = layer(self.activations[-1])
            if idx in eltwise:
                hidden = tf.add(hidden, self.activations[-2]) * 0.5
            self.activations.append(hidden) 
 
        self.outputs = self.activations[-1]
        
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess,  path + "temp/%s.ckpt" % self.name)


    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path =  path + "temp/%s.ckpt" % self.name
        saver.restore(sess, save_path)

class GCN(Model):
    
    # defination of GCN model
    def __init__(self, placeholders,**kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _loss(self):
        # loss function
        self.loss += mesh_loss(self.outputs, self.placeholders)

    def _build(self):
       
        self.build_unet()

        self.layers.append(GraphProjection(placeholders=self.placeholders))

        # GCN layers
        self.layers.append(GraphConvolution(input_dim=FLAGS.feat_dim,
                                            output_dim=FLAGS.hidden,  
                                            gcn_block_id=1,                                        
                                            placeholders=self.placeholders,
                                            logging=self.logging))
        for _ in range(8):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                                output_dim=FLAGS.hidden,   
                                                gcn_block_id=1,                                           
                                                placeholders=self.placeholders,
                                                logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden,
                                            output_dim=FLAGS.coord_dim,
                                            act=lambda x: x,   
                                            gcn_block_id=1,                                       
                                            placeholders=self.placeholders,
                                            logging=self.logging))
       

    def build_unet(self):
        x = self.placeholders['img_inp']
        x = tf.expand_dims(x, 0)
        
        #640 640
        x=tflearn.batch_normalization(x)
        x=tflearn.layers.conv.conv_2d(x,20,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,20,(3,3),strides=1,activation='relu',trainable=True)
        x0=x

        #320 320
        x=tflearn.max_pool_2d(x, (2, 2))
        x=tflearn.layers.conv.conv_2d(x,40,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,40,(3,3),strides=1,activation='relu',trainable=True)
        x1=x

        #160 160
        x=tflearn.max_pool_2d(x, (2, 2))
        x=tflearn.layers.conv.conv_2d(x,80,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,80,(3,3),strides=1,activation='relu',trainable=True)
        x2=x

        #80 80
        x=tflearn.max_pool_2d(x, (2, 2))
        x=tflearn.layers.conv.conv_2d(x,160,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,160,(3,3),strides=1,activation='relu',trainable=True)
        x3=x

        #40 40
        x=tflearn.max_pool_2d(x, (2, 2))
        x=tflearn.layers.conv.conv_2d(x,320,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,320,(3,3),strides=1,activation='relu',trainable=True)
        x4=x

        #20 20
        x=tflearn.max_pool_2d(x, (2, 2))
        x=tflearn.layers.conv.conv_2d(x,640,(3,3),strides=1,activation='relu',trainable=True)
        x=tflearn.layers.conv.conv_2d(x,640,(3,3),strides=1,activation='relu',trainable=True)
        x5=x

        u6 = tflearn.layers.conv.conv_2d_transpose(x5, 320, 2, [40, 40], strides=2, padding='same', name ='upsample_1',trainable=True)
        u6 = merge([u6, x4], mode='concat', axis=3, name='upsamle-1-merge')
        u6 = conv_2d(u6, 320, 3, activation='relu', name="conv6_1",trainable=True)
        x6 = conv_2d(u6, 320, 3, activation='relu', name="conv6_1",trainable=True)

        u7 = tflearn.layers.conv.conv_2d_transpose(x6, 160, 2, [80, 80], strides=2, padding='same', name ='upsample_2',trainable=True)
        u7 = merge([u7, x3], mode='concat', axis=3, name='upsamle-2-merge')
        u7 = conv_2d(u7, 160, 3, activation='relu', name="conv7_1",trainable=True)
        x7 = conv_2d(u7, 160, 3, activation='relu', name="conv7_1",trainable=True)

        u8 = tflearn.layers.conv.conv_2d_transpose(x7, 80, 2, [160, 160], strides=2, padding='same', name ='upsample_3',trainable=True)
        u8 = merge([u8, x2], mode='concat', axis=3, name='upsamle-3-merge')
        u8 = conv_2d(u8, 80, 3, activation='relu', name="conv8_1",trainable=True)
        x8 = conv_2d(u8, 80, 3, activation='relu', name="conv8_1",trainable=True)

        u9 = tflearn.layers.conv.conv_2d_transpose(x8, 40, 2, [320, 320], strides=2, padding='same', name ='upsample_4',trainable=True)
        u9 = merge([u9, x1], mode='concat', axis=3, name='upsamle-4-merge')
        u9 = conv_2d(u9, 40, 3, activation='relu', name="conv9_1",trainable=True)
        x9 = conv_2d(u9, 40, 3, activation='relu', name="conv9_1",trainable=True)

        u10 = tflearn.layers.conv.conv_2d_transpose(x9, 20, 2, [640, 640], strides=2, padding='same', name ='upsample_5',trainable=True)
        u10 = merge([u10, x0], mode='concat', axis=3, name='upsamle-5-merge')
        u10 = conv_2d(u10, 20, 3, activation='relu', name="conv10_1",trainable=True)
        x10 = conv_2d(u10, 20, 3, activation='relu', name="conv10_1",trainable=True)

        fc = tflearn.layers.conv.conv_2d(x10, 3, (1, 1), activation='linear',trainable=True)

        # update image feature and loss
        self.placeholders.update({'img_feat': [tf.squeeze(fc), tf.squeeze(x8), tf.squeeze(x9), tf.squeeze(x10)]})
        self.loss += tf.reduce_mean(tf.square(tf.subtract(tf.squeeze(fc), self.placeholders['img_label']))) / 6800.0
