"""
Atomik içerik ve fizikokimyasal özelliklerin bulunacağı sütunların isimleri,
sırasyla ATC_COL_NAMES ve PHYSICO_COL_NAMES değişkenlerine atanır.
"""
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
        
    physicochemical_properties_of_seq.append(analyze.molecular_weight())                #Molecular_Weight
    physicochemical_properties_of_seq.append(analyze.aromaticity())                     #Aromaticity
    physicochemical_properties_of_seq.append(analyze.instability_index())               #Instability_Index
    physicochemical_properties_of_seq.append(analyze.isoelectric_point())               #Isoelectric_Point
    physicochemical_properties_of_seq.append(analyze.gravy())                           #Hydrophaticity
    physicochemical_properties_of_seq.append(analyze.molar_extinction_coefficient()[0]) #Redyuced_Cysteines
    physicochemical_properties_of_seq.append(analyze.molar_extinction_coefficient()[1]) #Disulphate_Bonds
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

"""
Önceki aşamada temizlenmiş olan veri seti çağrılarak "cleaned_data_set" isimli değişkene atanır.
Amino asit içeriği, dipeptit içeriği, tripeptit içeriği, atomik içerik ve fizikokimyasal özellik
değerlerinin tutulacağı listeler oluşturulur.
"""
cleaned_data_set = pd.read_csv(r"../dataset/main_data_set(cleaned).csv", index_col = [0])
print(cleaned_data_set.shape)

all_aac = []
all_dpc = []
all_tpc = []
all_atc = []
all_physicochemical_properties = []

"""
Fizikokimyasal özellikler haricindeki öznitelikleri çıkartmak için Python'ın 'protlearn'
modülü içerisindeki;
    aac   ---> amino asit içerikleri
    ngram ---> dipeptit ve tripeptit içerikleri
    atc   ---> atomik içeriği
yöntemleri kullanılır.
İlgili yöntemler, özniteliklerin sayısal değerlerini çıkarttığı gibi, sütun isimlerini de
alfabetik sıraya göre çıkartmaktadır.
"""
from protlearn.features import aac, ngram, atc

"""
Temizlenmiş veri setindeki her bir diziye, ilgili öznitelik çıkarma yöntemleri
uygulanarak her dizinin değerleri liste olarak, ilgili özelliğin listesine eklenir.
Yani her dizinin tüm öznitelik değerlerini taşıyan listeler grubu, tek bir öznitelik yöntemi
listesinin içerisinde listelenecek şekilde bir iç içe liste yapısı oluşturulur.
"""
for seq in cleaned_data_set["SEQUENCE"]:
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

"""
Liste halindeki öznitelik grupları dataframe'lere çevrilir.
Temizlenmiş veri seti ile tüm öznitelik gruplarının oluşturdukları dataframe'ler
birleştirilerek tek bir veri seti haline getirilir ve 'extracted_data_set' isimli değişkene atanır.
"""
aac_dataframe = pd.DataFrame(np.array(all_aac), columns = list(aa_col_names))
dpc_dataframe = pd.DataFrame(np.array(all_dpc), columns = di_col_names)
tpc_dataframe = pd.DataFrame(np.array(all_tpc), columns = tri_col_names)
atc_dataframe = pd.DataFrame(np.array(all_atc), columns = ATC_COL_NAMES)
physico_dataframe = pd.DataFrame(np.array(all_physicochemical_properties), columns = PHYSICO_COL_NAMES)

extracted_data_set = cleaned_data_set.join([aac_dataframe, dpc_dataframe, tpc_dataframe, atc_dataframe, physico_dataframe])

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
Bazı öznitelikler birbirleri ile çok yüksek korelasyona sahip olabilir.
Makine öğrenmesi ve derin öğrenme gibi yöntemlerin eğitilmesi sürecinde çok
yüksek korelasyona sahip öznitelikler, birbirlerinin kopyası gibi davranıp
sonuçlarda yanlılığa sebep olabilirler.
Bu durumun önüne geçmek için, öznitelikler arasında bir korelasyon matrisi
oluşturulur ve 0.90'dan daha yüksek korelasyona sahip her iki öznitelikten bir
tanesi veri setinden çıkartılır.
Buna göre, aşırı yüksek korelasyona sahip 'GCG', 'PPV' tripeptitleri ve 
'AROMATICITY', 'EXTINCTION_COEFFICIENT_2' fizikokimyasal özellikleri veri 
setinden çıkartılır. 
"""
correlation_matrix = extracted_data_set.drop(["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"], axis = 1).corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k = 1).astype(np.bool))

highly_correlated = [c for c in upper_triangle.columns if any(upper_triangle[c] > 0.9)]
print(highly_correlated)

extracted_data_set = extracted_data_set.drop(highly_correlated, axis = 1)
print(extracted_data_set.shape)

"""
Son olarak, öznitelikleri çıkartılmış ve normalize edilmiş veri seti, csv olarak kaydedilir.
"""
extracted_data_set.to_csv("../dataset/main_data_set(extracted).csv")