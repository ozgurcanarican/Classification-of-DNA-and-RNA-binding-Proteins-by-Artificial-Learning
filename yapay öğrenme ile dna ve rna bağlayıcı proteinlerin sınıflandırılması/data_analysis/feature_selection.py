"""
Veri setini açmak için pandas kütüphanesi çağrılır.
"""
import pandas as pd

dataset = pd.read_csv(r"../dataset/main_data_set(extracted).csv", index_col = [0])
print(dataset.shape)

"""
Öznitelik seçme işlemi için ANOVA f test ile özniteliklerin sınıflarla olan ilişkisi
hesaplanacak ve en igili olan 'k' kadar öznitelik kalacak, diğerleri verisetinden
çıkartılacaktır. Bunun için öznitelikler 'features', sınıflar 'target' isimli
değişkenlere atanır.
"""
features = dataset.drop(["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"], axis = 1)
target = dataset["TYPE"]

print(features.shape)
print(target.shape)

"""
Öznitelik seçme için Sklearn kütüphanesinin SelectKBest metodu kullanılır. SelectKBest
'k' parametresi, yani kaç tane özniteliğin seçileceğini alır. ANOVA f-test uygular.
Başlangıç veriseti haricinde, 6000, 4000, 2000, 1000 ve 500 seçilmiş öznitelikle
ayrı ayrı verisetleri oluşturulur. Öznitelik sayısı her düşürüldüğünde veri
karmaşıklığı daha da azalmaktadır.
"""
from sklearn.feature_selection import SelectKBest

selector_six_k = SelectKBest(k = 6000)
selected_first_stage = selector_six_k.fit_transform(features, target)
six_k_col_names = selector_six_k.get_feature_names_out()
six_k_data = pd.concat([dataset[["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"]],
                       pd.DataFrame(selected_first_stage, columns = six_k_col_names)], axis = 1)

selector_four_k = SelectKBest(k = 4000)
selected_second_stage = selector_four_k.fit_transform(features, target)
four_k_col_names = selector_four_k.get_feature_names_out()
four_k_data = pd.concat([dataset[["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"]],
                        pd.DataFrame(selected_second_stage, columns = four_k_col_names)], axis = 1)

selector_two_k = SelectKBest(k = 2000)
selected_third_stage = selector_two_k.fit_transform(features, target)
two_k_col_names = selector_two_k.get_feature_names_out()
two_k_data = pd.concat([dataset[["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"]],
                        pd.DataFrame(selected_third_stage, columns = two_k_col_names)], axis = 1)

selector_one_k = SelectKBest(k = 1000)
selected_forth_stage = selector_one_k.fit_transform(features, target)
one_k_col_names = selector_one_k.get_feature_names_out()
one_k_data = pd.concat([dataset[["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"]],
                        pd.DataFrame(selected_forth_stage, columns = one_k_col_names)], axis = 1)

selector_five_h = SelectKBest(k = 500)
selected_fifth_stage = selector_five_h.fit_transform(features, target)
five_h_col_names = selector_five_h.get_feature_names_out()
five_h_data = pd.concat([dataset[["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"]],
                         pd.DataFrame(selected_fifth_stage, columns = five_h_col_names)], axis = 1)

"""
Öznitelik seçilerek oluşturulmuş verisetlerinin boyutları kontrol edilir.
"""
print(six_k_data.shape)
print(four_k_data.shape)
print(two_k_data.shape)
print(one_k_data.shape)
print(five_h_data.shape)

"""
Yeni verisetleri csv olarak kaydedilir.
"""
six_k_data.to_csv("../dataset/main_data_set(selected_6000).csv")
four_k_data.to_csv("../dataset/main_data_set(selected_4000).csv")
two_k_data.to_csv("../dataset/main_data_set(selected_2000).csv")
one_k_data.to_csv("../dataset/main_data_set(selected_1000).csv")
five_h_data.to_csv("../dataset/main_data_set(selected_500).csv")

