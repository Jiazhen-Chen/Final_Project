from tensorflow.keras.utils import to_categorical  
import pandas as pd
import numpy as np
import os

class Substrate(object):
    def __init__(self,number,name):
        self.number = number
        self.name = name

class Cofactor(object):
    def __init__(self,number,name):
        self.number = number
        self.name = name

class Enzyme(object):
    def __init__(self,number,name):
        self.number = number
        self.name = name

#Substrates
_3PG = Substrate(0,"3-PG")
_2PG = Substrate(1,"2-PG")
_PEP = Substrate(2,"PEP")
_PYR = Substrate(3,"PYR")

#Cofactors
_PPi = Cofactor(4,"PPi")
_AMP = Cofactor(5,"AMP")
_ATP = Cofactor(6,"ATP")
_Pi = Cofactor(7,"Pi")

#Enzyme data
excel_path = os.path.join(os.path.join(os.path.dirname(__file__), "data"),"enzyme_activity_data.xlsx")
data = pd.read_excel(excel_path, sheet_name=0)

new_path = os.path.join(os.path.dirname(__file__), "converted_data")
if not os.path.exists(new_path):
    os.makedirs(new_path)

data.to_csv(os.path.join(new_path,"enzyme_activity_data.csv"))

PGAM_Activity = pd.DataFrame(data,columns=["PGAM(uM)"]).to_numpy()
ENO_Activity = pd.DataFrame(data,columns=["ENO(uM)"]).to_numpy()
PPDK_Activity = pd.DataFrame(data,columns=["PPDK(uM)"]).to_numpy()
flux_data = pd.DataFrame(data,columns=["J(nmol/min)"]).to_numpy()

#Enzymes
_PGAM = Enzyme(8,"PGAM")
_ENO = Enzyme(9,"ENO")
_PDDK = Enzyme(10,"PDDK")

#Dictionary
component_list = []
#Substrate
component_list.append([_3PG.number,_3PG.name,type(_3PG)])
component_list.append([_2PG.number,_2PG.name,type(_2PG)])
component_list.append([_PEP.number,_PEP.name,type(_PEP)])
component_list.append([_PYR.number,_PYR.name,type(_PYR)])
#Cofactor
component_list.append([_PPi.number,_PPi.name,type(_PPi)])
component_list.append([_AMP.number,_AMP.name,type(_AMP)])
component_list.append([_ATP.number,_ATP.name,type(_ATP)])
component_list.append([_Pi.number,_Pi.name,type(_Pi)])
#Enzymes
component_list.append([_PGAM.number,_PGAM.name,type(_PGAM)])
component_list.append([_ENO.number,_ENO.name,type(_ENO)])
component_list.append([_PDDK.number,_PDDK.name,type(_PDDK)])

component_df = pd.DataFrame(component_list, columns = ['Number', 'Name','Class'])
component_npy = component_df.to_numpy()
class_type = pd.DataFrame(component_df,columns=["Class"]).to_numpy()

#convert class_type to onehot encoding
n = 0
for entry in class_type:
    
    if entry == Substrate:
        class_type[n] = 0

    if entry == Cofactor:
        class_type[n] = 1
    
    if entry == Enzyme:
        class_type[n] = 2
    
    n = n+1

class_type_lables = to_categorical(class_type, num_classes=3)


np.save(os.path.join(new_path,"component_npy.npy"),component_npy[:,0:2])
np.save(os.path.join(new_path,"class_type_labels.npy"),class_type_lables)
np.save(os.path.join(new_path,"PGMA_Activity.npy"),PGAM_Activity)
np.save(os.path.join(new_path,"ENO_Activity.npy"),ENO_Activity)
np.save(os.path.join(new_path,"PPDK_Activity.npy"),PPDK_Activity)
np.save(os.path.join(new_path,"flux_data.npy"),flux_data)


