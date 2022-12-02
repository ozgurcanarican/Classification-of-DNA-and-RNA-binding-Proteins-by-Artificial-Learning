ATC_COL_NAMES = ["C ATOM", "H ATOM", "N ATOM", "O ATOM", "S ATOM"]

PHYSICO_COL_NAMES = ["MOLECULAR_WEIGHT", "AROMATICITY", "INSTABILITY_INDEX", "ISOELECTRIC_POINT",
                     "GRAVY", "EXTINCTION_COEFFICIENT_1", "EXTINCTION_COEFFICIENT_2",
                     "HELIX_STRUCTURE", "TURN_STRUCTURE", "SHEET_STRUCTURE"]


def physicochemical_properties(seq):
    
    """
    'physicochemical_properties' fonksiyonu dizilerin çeşitli fizikokimyasal
    özelliklerinin sayısal değerlerini hesaplar. Bunun için Biopython kütüphanesinden
    ProteinAnalysis (PA) metodu kullanılır.
    
    Fonksiyon 'seq' parametresini yani protein dizi bilgisini alır,
    PA yardımıyla ilgili fizikokimyasal özelliğin sayısal değerini hesaplar.
    
    En sonunda 10 farklı fizikokimyasal özelliğin sayısal değerinin bulunduğu
    bir liste döndürür.
    """
    
    physicochemical_properties_of_seq = []
        
    analyze = PA(seq)
        
    physicochemical_properties_of_seq.append(analyze.molecular_weight())
    physicochemical_properties_of_seq.append(analyze.aromaticity())
    physicochemical_properties_of_seq.append(analyze.instability_index())
    physicochemical_properties_of_seq.append(analyze.isoelectric_point())
    physicochemical_properties_of_seq.append(analyze.gravy())                           #Hydrophaticity
    physicochemical_properties_of_seq.append(analyze.molar_extinction_coefficient()[0]) #Redyuced Cysteines
    physicochemical_properties_of_seq.append(analyze.molar_extinction_coefficient()[1]) #Disulphate Bonds
    physicochemical_properties_of_seq.append(analyze.secondary_structure_fraction()[0]) #Helix
    physicochemical_properties_of_seq.append(analyze.secondary_structure_fraction()[1]) #Turn
    physicochemical_properties_of_seq.append(analyze.secondary_structure_fraction()[2]) #Sheet
    
    return physicochemical_properties_of_seq


"""
Matris, dizi ve listeler ile çalışılacağından, Numpy ve Pandas kütüphaneleri çağrılır.
Fizikokimyasal özellikleri hesaplamak için Biopython'dan ProteinAnalysis aracı çağrılır.
"""
import numpy as np    
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA

test474 = pd.read_csv(r"../alternative_dataset/test474(main).csv", index_col = [0])
pdb255 = pd.read_csv(r"../alternative_dataset/pdb255(main).csv", index_col = [0])

print(test474.shape)
print(pdb255.shape)

dataset = [test474, pdb255]
dataset_name = ["test474", "pdb255"]

for d in range(len(dataset)):

    all_aac = []
    all_dpc = []
    all_tpc = []
    all_atc = []
    all_physicochemical_properties = []
    
    from protlearn.features import aac, ngram, atc
    
    for seq in dataset[d]["SEQUENCE"]:
        aa_comp, aa_col_names = aac(seq, remove_zero_cols = False)
        [aa_comp] = aa_comp.tolist()
        all_aac.append(aa_comp)
        
        di_comp, di_col_names = ngram(seq, n = 2)
        [di_comp] = di_comp.tolist()
        all_dpc.append(di_comp)
        
        tri_comp, tri_col_names = ngram(seq, n = 3)
        [tri_comp] = tri_comp.tolist()
        all_tpc.append(tri_comp)
        
        atom_comp, bonds = atc(seq)
        [atom_comp] = atom_comp.tolist()
        all_atc.append(atom_comp)
        
        all_physicochemical_properties.append(physicochemical_properties(seq))
    
    aac_dataframe = pd.DataFrame(np.array(all_aac), columns = list(aa_col_names))
    dpc_dataframe = pd.DataFrame(np.array(all_dpc), columns = di_col_names)
    tpc_dataframe = pd.DataFrame(np.array(all_tpc), columns = tri_col_names)
    atc_dataframe = pd.DataFrame(np.array(all_atc), columns = ATC_COL_NAMES)
    physico_dataframe = pd.DataFrame(np.array(all_physicochemical_properties), columns = PHYSICO_COL_NAMES)
    
    extracted_data_set = dataset[d].join([aac_dataframe, dpc_dataframe, tpc_dataframe, atc_dataframe, physico_dataframe])
    
    """   
        DATA NORMALIZATION
        
    AAC, DPC gibi yöntemler, protein birimlerinin fraksiyonları ile ilgilendiğinden,
    doğal olarak 0 ve 1 arasındaki değerleri döndürmektedirler.
    
    Buna karşılık bazı fizikokimyasal özellikler, negatif veya 0'dan büyük sayılar
    döndürebilirler. Verilerin büyük bir çoğunluğu fraksiyonel olduğundan, min-max
    ölçeklendirme yöntemi ile fraksiyonel olmayan sütunların normalize edilmesi uygun
    bulunmuştur.
    
    Bu bağlamda, normalize edilmesi gereken sütunlar;
    
        -Molecular weight
        -Instability index
        -Isoelectric point
        -Gravy
        -Extinction_coefficient x1
        -Extinction_coefficient x2
        
    olarak belirlenmiştir.
    
    Normalizasyon işlemi için sklearn kütüphanesinin, MinMaxScaler metodu kullanılmaktadır.
    """
    
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    extracted_data_set[["MOLECULAR_WEIGHT", 
                        "INSTABILITY_INDEX",
                        "ISOELECTRIC_POINT",
                        "GRAVY", 
                        "EXTINCTION_COEFFICIENT_1", 
                        "EXTINCTION_COEFFICIENT_2"]] = scaler.fit_transform(extracted_data_set[["MOLECULAR_WEIGHT", 
                                                                                                "INSTABILITY_INDEX", 
                                                                                                "ISOELECTRIC_POINT",
                                                                                                "GRAVY", 
                                                                                                "EXTINCTION_COEFFICIENT_1", 
                                                                                                "EXTINCTION_COEFFICIENT_2"]])
    
    """
    'GCG', 'PPV' tripeptitleri ve 'AROMATICITY', 'EXTINCTION_COEFFICIENT_2'
    fizikokimyasal özellikleri, orijinal veri setinde birbirine çok yakın korelasyona 
    sahip olduklarından çıkartılmıştır. Aynı öznitelikler alternatif verisetlerinde de çıkartılmalıdır.
    """
    extracted_data_set = extracted_data_set.drop(['GCG', 'PPV', 'AROMATICITY', "EXTINCTION_COEFFICIENT_2"], axis = 1)
    
    """
    Altertatif veri setinde "DATABASE_ID", orijinalinde "UNIPROT_ID" olarak geçtiği için
    bu sütunu atıyor. Alternatif veri setlerindeki ID isimleri değiştirilir.
    """
    extracted_data_set.rename({"DATABASE_ID" : "UNIPROT_ID"}, axis = 1, inplace = True)
    
    extracted_data_set.to_csv("../alternative_dataset/{}(extracted).csv".format(dataset_name[d]))
