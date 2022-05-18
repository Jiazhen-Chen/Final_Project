import numpy as np
import os
import pickle

data_path = os.path.join(os.path.dirname(__file__), "converted_data")
class_type_labels = np.load(os.path.join(data_path,"class_type_labels.npy"),allow_pickle=True)
ENO_Activity = np.load(os.path.join(data_path,"ENO_Activity.npy"),allow_pickle=True)
PPDK_Activity = np.load(os.path.join(data_path,"PPDK_Activity.npy"),allow_pickle=True)
PGMA_Activity = np.load(os.path.join(data_path,"PGMA_Activity.npy"),allow_pickle=True)
flux_data = np.load(os.path.join(data_path,"flux_data.npy"),allow_pickle=True)
component_array = np.load(os.path.join(data_path,"component_npy.npy"),allow_pickle=True)

pair_indices = [[0,8],[0,9],[1,8],[1,9],
                [2,9],[2,10],[3,10],[4,8],
                [4,9],[4,10],[5,10],[6,10],
                [7,10],[8,0],[8,1],[9,1],                
                [9,2],[10,2],[10,3],[10,4],
                [10,5],[10,6],[10,7]]

pair_name = ["3PG_PGAM_fwd","3PG_ENO_inhb","2PG_PGAM_rev","2PG_ENO_fwd",
             "PEP_ENO_rev","PEP_PPDK_fwd","Pyr_PPDK_rev","PPi_PGAM_inhb",
             "PPi_ENO_inhb","PPi_PPDK_fwd","AMP_PPDK_fwd","ATP_PPDK_rev",
             "Pi_PPDK_rev","PGAM_3PG_rev","PGAM_2PG_fwd","ENO_2PG_rev",
             "ENO_PEP_fwd","PPDK_PEP_rev","PPDK_Pyr_fwd","PPDK_PPi_rev",
             "PPDK_AMP_rev","PPDK_ATP_fwd","PPDK_Pi_fwd"]
 
# in the form of [Km,vmax,Ki]
edge_features = [[473,75,0],[0,0,610],[106,67.24,0],[86.4,328.5,0],
                [102,66.61,0],[30,196.5,0],[221,12.28,0],[0,0,173],
                [0,0,137],[91,196.5,0],[2,196.5,0],[597,12.28,0],
                [1342,12.28,0],[473,67.24,0],[106,75,0],[86.4,66.61,0],
                [102,328.5,0],[30,12.28,0],[221,196.5,0],[91,12.28,0],
                [2,12.28,0],[597,196.5,0],[1342,196.5,0]]

#Generate node data
node_3PG = np.zeros(len(flux_data),dtype=object)
node_2PG = np.zeros(len(flux_data),dtype=object)
node_PEP = np.zeros(len(flux_data),dtype=object)
node_PYR = np.zeros(len(flux_data),dtype=object)
node_AMP = np.zeros(len(flux_data),dtype=object)
node_PPi = np.zeros(len(flux_data),dtype=object)
node_ATP = np.zeros(len(flux_data),dtype=object)
node_Pi = np.zeros(len(flux_data),dtype=object)
node_PGAM = np.zeros(len(flux_data),dtype=object)
node_ENO = np.zeros(len(flux_data),dtype=object)
node_PPDK = np.zeros(len(flux_data),dtype=object)

for i in range(len(flux_data)):
    node_3PG[i] = np.concatenate((np.array([4000]),np.array(class_type_labels[0])),axis=0) #Lo-Thong
    node_2PG[i] = np.concatenate((np.array([142]),np.array(class_type_labels[1])),axis=0)#Moreno-Sánchez
    node_PEP[i] = np.concatenate((np.array([24.5]),np.array(class_type_labels[2])),axis=0)#Moreno-Sánchez
    node_PYR[i] = np.concatenate((np.array([60]),np.array(class_type_labels[3])),axis=0) #bionumbers 101233
    node_AMP[i] = np.concatenate((np.array([200]),np.array(class_type_labels[5])),axis=0) #Lo-Thong
    node_PPi[i] = np.concatenate((np.array([1700]),np.array(class_type_labels[4])),axis=0) #Lo-Thong
    node_ATP[i] = np.concatenate((np.array([3000]),np.array(class_type_labels[6])),axis=0) #Lo-Thong
    node_Pi[i] = np.concatenate((np.array([4000]),np.array(class_type_labels[7])),axis=0) #Lo-Thong
    node_PGAM[i] = np.concatenate((np.array(PGMA_Activity[i]),np.array(class_type_labels[8])),axis=0) #Lo-Thong
    node_ENO[i] = np.concatenate((np.array(ENO_Activity[i]),np.array(class_type_labels[9])),axis=0) #Lo-Thong
    node_PPDK[i] = np.concatenate((np.array(PPDK_Activity[i]),np.array(class_type_labels[10])),axis=0) #Lo-Thong
    
node_features_list = np.zeros((len(flux_data),len(component_array),len(node_3PG[0])),dtype=float)
edge_features_list = np.zeros((len(flux_data),len(edge_features),len(edge_features[0])),dtype=float)
pair_indices_list = np.zeros((len(flux_data),len(pair_indices),len(pair_indices[0])),dtype=int)

for i in range(len(flux_data)):
    node_features_list[i] = np.concatenate(([node_3PG[i]],[node_2PG[i]],[node_PEP[i]],
                                            [node_PYR[i]],[node_AMP[i]],[node_PPi[i]],
                                            [node_ATP[i]],[node_Pi[i]],[node_PGAM[i]],
                                            [node_ENO[i]],[node_PPDK[i]]
                                            ),axis=0)
    
    edge_features_list[i] = np.array((edge_features),dtype=float)
    pair_indices_list[i] = np.array((pair_indices),dtype=int)

new_path = os.path.join(os.path.dirname(__file__), "NN_data")
if not os.path.exists(new_path):
    os.makedirs(new_path)
#shuffle data before training to prevent influcence of data order
shuffle = np.random.permutation(np.arange(len(flux_data)))
train = shuffle[:int(len(shuffle)*0.8)]
#generate training and validation data
node_features_list_train = node_features_list[train]
edge_features_list_train = edge_features_list[train]
pair_indices_list_train = pair_indices_list[train]
flux_data_train = flux_data[train]

node_features_list_val = node_features_list[int(len(shuffle)*0.8):len(flux_data)]
edge_features_list_val = edge_features_list[int(len(shuffle)*0.8):len(flux_data)]
pair_indices_list_val = pair_indices_list[int(len(shuffle)*0.8):len(flux_data)]
flux_data_val = flux_data[int(len(shuffle)*0.8):len(flux_data)]

x_train = node_features_list_train, edge_features_list_train, pair_indices_list_train
y_train = flux_data_train

x_val =  node_features_list_val, edge_features_list_val, pair_indices_list_val
y_val = flux_data_val

#save generated training and validation data
np.save(os.path.join(new_path,"node_features_list_train.npy"),node_features_list_train)
np.save(os.path.join(new_path,"edge_features_list_train.npy"),edge_features_list_train)
np.save(os.path.join(new_path,"pair_indices_list_train.npy"),pair_indices_list_train)
np.save(os.path.join(new_path,"flux_data_train.npy"),flux_data_train)

np.save(os.path.join(new_path,"node_features_list_val.npy"),node_features_list_val)
np.save(os.path.join(new_path,"edge_features_list_val.npy"),edge_features_list_val)
np.save(os.path.join(new_path,"pair_indices_list_val.npy"),pair_indices_list_val)
np.save(os.path.join(new_path,"flux_data_val.npy"),flux_data_val)

np.save(os.path.join(new_path,"component_npy.npy"),component_array)

pickle.dump(x_train,open(os.path.join(new_path,"x_train.p"),"wb"))
pickle.dump(y_train,open(os.path.join(new_path,"y_train.p"),"wb"))

pickle.dump(x_val,open(os.path.join(new_path,"x_val.p"),"wb"))
pickle.dump(y_val,open(os.path.join(new_path,"y_val.p"),"wb"))


