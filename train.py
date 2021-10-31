import tensorflow.compat.v1 as tf
from model import GCN
import numpy as np

from fetcher import *
from utils import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("train_list", "../Data/train_list.txt", "train list.")
flags.DEFINE_string("test_list", "../Data/test_list.txt", "test list.")
flags.DEFINE_float("learning_rate", 1e-4, "Initial learning rate.")
flags.DEFINE_float("dropout", 0.5, "Dropout rate (1 - keep probability).") #not used
flags.DEFINE_integer("epochs", 300, "Number of epochs to train.")
flags.DEFINE_integer("hidden",256, "Number of units in hidden layer.") 
flags.DEFINE_integer("feat_dim", 6, "Number of units in feature layer.")
flags.DEFINE_integer("coord_dim", 3, "Number of units in output layer.")
flags.DEFINE_float("weight_decay", 5e-6, "Weight decay for L2 loss.")

def train():

    # Specify target organ
    organ = "liver"
    nNode = 500
                
    # Set random seed
    seed = 1024
    np.random.seed(seed)
    tf.set_random_seed(seed)

    num_supports = 1

    # Define placeholders(dict) and model
    placeholders = {
        "features": tf.placeholder(tf.float32, shape=(None, 3)),        # initial 3D coordinates
        "labels": tf.placeholder(tf.float32, shape=(None, 3)),          # ground truth
        "img_inp": tf.placeholder(tf.float32, shape=(640, 640, 3)),     # initial projection + DRR
        "img_label": tf.placeholder(tf.float32, shape=(640, 640, 3)),   # target deformation map
        "shapes": tf.placeholder(tf.float32, shape=(None, 3)),          # relative positions
        "ipos": tf.placeholder(tf.float32, shape=(None, 2)),            # initial projected points
        "adj" : tf.placeholder(tf.float32, shape=(nNode,nNode)),        # adj matrix size
        "rmax": tf.placeholder(tf.float32),                             # rmax for projection        
        "face": tf.placeholder(tf.int32, shape=(None, 4)),              # triangle face
        "face_norm": tf.placeholder(tf.float32, shape=(None, 3)),       # face normal vector
        "support": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)], # for convolution
        "num_features_nonzero": tf.placeholder(tf.int32),                            # helper variable for sparse dropout
    }

    # Create model
    model = GCN(placeholders, logging=True)

    # Initialize session
    data = DataFetcher(FLAGS.train_list, organ)
    data.setDaemon(True)
    data.start()

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    train_number = data.number
    print("[X2S] %s, train data:%d \n" % (organ, train_number), end="", flush=True)

    # Train model
    for epoch in range(FLAGS.epochs):
        all_loss = np.zeros(train_number, dtype="float32")
        all_errors = np.zeros(train_number, dtype="float32")

        for iters in range(train_number):

            adj, features, labels, trans, face_norm, face, rmax, projM, img_inp, img_label, shapes, ipos, caseID, phaseID = data.fetch()  
            support = [preprocess_adj(adj)]

            # Update placeholder
            feed_dict = construct_feed_dict(features, labels, img_inp, img_label, shapes, ipos, adj, rmax, face, face_norm, support, placeholders)

            # Training and update weights
            _, dists, outs = sess.run([model.opt_op, model.loss, model.outputs], feed_dict=feed_dict)

            features = features * rmax 
            labels = labels * rmax
            outs = outs * rmax

            for i in range(nNode):
                outs[i] = features[i] + outs[i]

            all_loss[iters] = dists
            all_errors[iters] = np.mean(np.sqrt(np.sum(np.square(np.subtract(outs, labels)), 1)))
            initial = np.mean(np.sqrt(np.sum(np.square(np.subtract(features, labels)), 1)))

            print("\r[Training:%d/%d] Init:%.3f mm, Error:%.3f mm" % (iters+1, train_number, initial, all_errors[iters]), end=" ", flush=True)

        # Save model
        if epoch % 100 == 0 :
            model.name = "{}".format(epoch)
            model.save(sess)
            model.name = "gcn"

        # print error metric
        mean_error = np.mean(all_errors[np.where(all_errors)])
        mean_loss = np.mean(all_loss[np.where(all_loss)])

        print("\r[Epoch:%d] Loss:%.4f, Error:%.3f mm                  \n" % (epoch+1, mean_loss, mean_error), end="", flush = True)

    data.shutdown()
    print ("Training Finished")


if __name__ == "__main__":

    train()


