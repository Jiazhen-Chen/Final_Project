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

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + tf.cast(increment[:, tf.newaxis],tf.int32)
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch).prefetch(-1)

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 0])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 1],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated
    
class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):

        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)
    
def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="node_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="edge_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    pathway_indicator = layers.Input((), dtype="int32", name="pathway_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, pathway_indicator])
    
    x = layers.Dense(dense_units, activation="relu",name="dense1")(x)

    x = layers.Dense(1, activation="relu",name="dense2")(x)

    model = tf.keras.Model(
        inputs=[atom_features, bond_features, pair_indices, pathway_indicator],
        outputs=[x],
    )
    return model


mpnn = MPNNModel(
    atom_dim=x_train_tf[0][0][0].shape[0], bond_dim=x_train_tf[1][0][0].shape[0],
)

mpnn.compile(optimizer = Adam(), 
             loss='mean_absolute_percentage_error',
             metrics = 'mean_absolute_percentage_error'
             )

tf.keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

train_data = MPNNDataset(x_train_tf, y_train_tf)
val_data = MPNNDataset(x_val_tf, y_val_tf)

model_fit = mpnn.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    verbose=2
    )

fig = plt.figure(figsize=(10, 6))
plt.plot(model_fit.history["mean_absolute_percentage_error"], label="train error")
plt.plot(model_fit.history["val_mean_absolute_percentage_error"], label="validation error")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Mean Absolute Percentage Error", fontsize=16)
plt.yscale("log")
plt.title("Training and Validation Error for Direct Implementation", fontsize=16, pad=20)
plt.legend(fontsize=16)
plt.show()

