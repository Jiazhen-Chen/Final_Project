import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


#convert pickle data to tensordata for tensorflow implementation
data_path = os.path.join(os.path.dirname(__file__), "NN_data")

component_array = np.load(os.path.join(data_path,"component_npy.npy"),allow_pickle=True)

x_train = pickle.load(open(os.path.join(data_path,"x_train.p"),"rb"))
node_features_train = tf.ragged.constant(x_train[0])
edge_features_train = tf.ragged.constant(x_train[1])
pair_indices_train = tf.ragged.constant(x_train[2])

y_train = pickle.load(open(os.path.join(data_path,"y_train.p"),"rb"))

flux_train = tf.constant(y_train)

x_val = pickle.load(open(os.path.join(data_path,"x_val.p"),"rb"))
node_features_val = tf.ragged.constant(x_val[0])
edge_features_val = tf.ragged.constant(x_val[1])
pair_indices_val = tf.ragged.constant(x_val[2])

y_val = pickle.load(open(os.path.join(data_path,"y_val.p"),"rb"))
flux_val = tf.constant(y_val)

x_train_tf = node_features_train,edge_features_train,pair_indices_train
y_train_tf = flux_train

x_val_tf = node_features_val,edge_features_val,pair_indices_val
y_val_tf = flux_val


def prepare_batch(x,y):
    node_features,edge_features,pair_indices = x
    
    node_features = node_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    edge_features = edge_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
    return (node_features,edge_features,pair_indices),y

def MPNNDataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    return dataset.batch(1).map(prepare_batch, -1).prefetch(-1)

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.node_dim = 4
        self.edge_dim = 3
        self.kernel = self.add_weight(
            shape=(self.edge_dim, self.node_dim * self.node_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.node_dim * self.node_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        node_features, edge_features, pair_indices = inputs
        
        # Apply linear transformation to edge features
        edge_features = tf.matmul(edge_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        edge_features = tf.reshape(edge_features, (-1, self.node_dim, self.node_dim))

        # Obtain features of neighbor nodes
        node_features_origin = tf.gather(node_features, pair_indices[:, 0])
        node_features_origin = tf.expand_dims(node_features_origin, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(edge_features, node_features_origin)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 1],
            num_segments=tf.shape(node_features)[0],
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, steps=2, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps

    def build(self, input_shape):
        self.node_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.update_step = layers.GRUCell(self.node_dim)
        self.built = True

    def call(self, inputs):
        node_features, edge_features, pair_indices = inputs
        node_features_updated = node_features
        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            node_features_aggregated = self.message_step(
                [node_features_updated, edge_features, pair_indices]
            )

            # Update node state via a step of GRU
            node_features_updated, _ = self.update_step(
                node_features_aggregated, node_features_updated
            )
        return node_features_updated

class ExtractNode(layers.Layer):    
    def __init__(self, enzyme_node_label=[8,9,10], **kwargs):
        super().__init__(**kwargs)
        self.enzyme_node_label = enzyme_node_label
        
    def call(self,inputs):
        x = tf.gather(inputs, self.enzyme_node_label)
        x = tf.expand_dims(tf.reshape(x,[-1]),axis= 0)
        return x
        
    
def MPNNmodel(node_dim,
              edge_dim,
              message_steps = 2,
              hidden_layer_units = 10,
              enzyme_node_label = [8,9,10]
              ):
    
    node_features = layers.Input((node_dim), dtype="float64", name="node_features")
    edge_features = layers.Input((edge_dim), dtype="float64", name="edge_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")

    
    x = MessagePassing(message_steps)([node_features, edge_features, pair_indices])
    
    x = ExtractNode(enzyme_node_label)(x)
      
    x = layers.Dense(hidden_layer_units, activation="sigmoid",name="dense1")(x)
    x = layers.Dense(1, activation="relu",name="dense2")(x)
    
    model = tf.keras.Model(
        inputs=[node_features, edge_features, pair_indices],
        outputs=[x]
        )
    return model

mpnn = MPNNmodel(node_dim=x_train_tf[0][0][0].shape[0],edge_dim = x_train_tf[1][0][0].shape[0])

mpnn.compile(optimizer = Adam(), 
             loss='mean_absolute_percentage_error',
             metrics = 'mean_absolute_percentage_error'
             )

tf.keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

train_data = MPNNDataset(x_train_tf, y_train_tf)
val_data = MPNNDataset(x_val_tf, y_val_tf)

model_fit = mpnn.fit(train_data,
                     validation_data = val_data,
                     epochs = 30,
                     verbose = 2,
                     )

fig = plt.figure(figsize=(10, 6))
plt.plot(model_fit.history["mean_absolute_percentage_error"], label="train error")
plt.plot(model_fit.history["val_mean_absolute_percentage_error"], label="validation error")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Mean Absolute Percentage Error", fontsize=16)
plt.yscale("log")
plt.title("Training and Validation Error for Modified Implementation", fontsize=16, pad=20)
plt.legend(fontsize=16)
plt.show()

