import pandas as pd

test474 = pd.read_csv(r"../alternative_dataset/test474(extracted).csv", index_col = [0])
pdb255 = pd.read_csv(r"../alternative_dataset/pdb255(extracted).csv", index_col = [0])

alternative_dataset = [test474, pdb255]
alternative_dataset_name = ["test474", "pdb255"]

initial_dataset = pd.read_csv(r"../dataset/main_data_set(extracted).csv", index_col = [0])
six_k_data = pd.read_csv(r"../dataset/main_data_set(selected_6000).csv", index_col = [0])
four_k_data  = pd.read_csv(r"../dataset/main_data_set(selected_4000).csv", index_col = [0])
two_k_data  = pd.read_csv(r"../dataset/main_data_set(selected_2000).csv", index_col = [0])
one_k_data = pd.read_csv(r"../dataset/main_data_set(selected_1000).csv", index_col = [0])
five_h_data = pd.read_csv(r"../dataset/main_data_set(selected_500).csv", index_col = [0])

print(six_k_data.shape)
print(four_k_data.shape)
print(two_k_data.shape)
print(one_k_data.shape)
print(five_h_data.shape)

for d in range(len(alternative_dataset)):

    six_k_data_alternative = alternative_dataset[d].filter(six_k_data.columns, axis = 1)
    print(six_k_data_alternative.shape)
    
    four_k_data_alternative = alternative_dataset[d].filter(four_k_data.columns, axis = 1)
    print(four_k_data_alternative.shape)
    
    two_k_data_alternative = alternative_dataset[d].filter(two_k_data.columns, axis = 1)
    print(two_k_data_alternative.shape)
    
    one_k_data_alternative = alternative_dataset[d].filter(one_k_data.columns, axis = 1)
    print(one_k_data_alternative.shape)
    
    five_h_data_alternative = alternative_dataset[d].filter(five_h_data.columns, axis = 1)
    print(five_h_data_alternative.shape)
    
    
    six_k_data_alternative.to_csv("../alternative_dataset/{}(selected_6000).csv".format(alternative_dataset_name[d]))
    four_k_data_alternative.to_csv("../alternative_dataset/{}(selected_4000).csv".format(alternative_dataset_name[d]))
    two_k_data_alternative.to_csv("../alternative_dataset/{}(selected_2000).csv".format(alternative_dataset_name[d]))
    one_k_data_alternative.to_csv("../alternative_dataset/{}(selected_1000).csv".format(alternative_dataset_name[d]))
    five_h_data_alternative.to_csv("../alternative_dataset/{}(selected_500).csv".format(alternative_dataset_name[d]))
