"""
Veri setini açmak için pandas kütüphanesi çağrılır.
"""
import pandas as pd

raw_data_set = pd.read_csv(r"../dataset/main_data_set(raw).csv", index_col = [0])
print(raw_data_set.shape)

"""
50 amino asit uzunluğundan daha kısa olan proteinler veri setinden çıkartılır.
1000 amino asit uzunluğundan daha uzun olan proteinler veri setinden çıkartılır.
"""
raw_data_set = raw_data_set[raw_data_set["LENGTH"] > 50]
raw_data_set = raw_data_set[raw_data_set["LENGTH"] < 1000]
print(raw_data_set.shape)

"""
Amino asitlerin 20 harflik gösterimi olan [A, R, N, D, C, Q, E, G, H, I, L, K,
M, F, P, S, T, W, Y, V] haricinde karakter barındıran proteinler veri setinden
çıkartılır.
Bunun için iç içe iki döngü oluşturulur. İlk döngü sütun içerisindeki her bir
proteini, ikinci döngü her bir protein içerisindeki amino asidi döndürür. Eğer,
dönen amino asitlerden herhangi biri 20'lik gösterimden farklı bir karakter ile
ifade edilmiş ise, içerdiği sekans veri setinden çıkartılır ve iç döngü 
durdurularak sıradaki sekansa geçilir.
"""
aaList = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
for seq in raw_data_set["SEQUENCE"]:
    seqIndex = 0
    isSequenceCorrect = True
    while seqIndex <= (len(seq) - 1) and isSequenceCorrect:
        if not seq[seqIndex] in aaList:
            raw_data_set = raw_data_set[raw_data_set["SEQUENCE"] != seq]
            isSequenceCorrect = False
        seqIndex += 1
print(raw_data_set.shape)

"""
Veriseti içerisinde, farklı kimliklere ait, aynı protein dizileri olabilir.
Birebir aynı dizilere sahip proteinler verisetinden çıkartılır.
"""
raw_data_set.drop_duplicates(subset = ["SEQUENCE"], inplace = True)

print("DBP_size:", raw_data_set[raw_data_set["TYPE"] == "dbp"].shape)
print("RBP_size:", raw_data_set[raw_data_set["TYPE"] == "rbp"].shape)
print("NNABP_size:", raw_data_set[raw_data_set["TYPE"] == "nnabp"].shape)
print("Dataset_size:", raw_data_set.shape)

"""
Bu noktaya kadar uygulanan tüm veri temizleme işlemlerinden sonra, toplamda
365851 veri kalmıştır. Bu kadar çok veri ile benzerlik algoritması çok yavaş
çalışacağından, her bir sınıftan 15000 veri olmak üzere toplamda 45000 veri
'sample' metodu ile rastgele seçilecektir.
"""
dbp_raw_data = raw_data_set[raw_data_set["TYPE"] == "dbp"].sample(n = 15000)
rbp_raw_data = raw_data_set[raw_data_set["TYPE"] == "rbp"].sample(n = 15000)
nnabp_raw_data = raw_data_set[raw_data_set["TYPE"] == "nnabp"].sample(n = 15000)

raw_data_set = pd.concat([dbp_raw_data, rbp_raw_data, nnabp_raw_data]).reset_index(drop = True)
print("Dataset_size:", raw_data_set.shape)

"""
Benzerlik(Identity) oranı %30'dan yüksek olan proteinler veri setinden çıkartılır.
Benzerlik hesaplamak için öncelikle protein dizileri hizalanmalıdır. Hizalama
için Biopython içerisindeki 'Align' metodu kullanılır.

Tüm proteinlerin birbirleri ile hizalanıp benzer olanları ayırabilmek için iç
içe iki for döngü yapısı kurulmuştur. İlk döngü sütun içerisindeki query dizisini
tutar, ikinci döngü aynı sütundaki, hizalanacak subject dizilerinin iterasyonunu
yapar. İkinci for içersindeki if kontrolü sayesinde her zaman subject konumundaki
dizinin endeks değerinin query'den büyük olması sağlanmıştır. Böylelikle ilk döngü
her seferinde başa döndüğünde, zaten hizalanmış olan proteinlerin tekrar hizalanıp
benzerlik değerinin hesaplanmasının önüne geçilmiştir.

Benzerlik değeri = tam eşleşme sağlayan amino asit sayısı / hizalanmış dizi uzunluğu
olarak hesaplanmaktadır.
"""

from Bio.Align import PairwiseAligner, substitution_matrices

sequences = list(raw_data_set["SEQUENCE"])

for query in sequences:
    for subject in sequences:
        if sequences.index(query) < sequences.index(subject):

            aligner = PairwiseAligner(mode = "local")
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            
            alignments = aligner.align(query, subject)
            one_alignment = str(alignments[0])
            
            match = one_alignment.splitlines()[1].count("|")
            aligned_length = len(one_alignment.splitlines()[1])
                        
            identity = match / aligned_length

            if identity > 0.30:
                sequences.remove(subject)
                raw_data_set = raw_data_set[raw_data_set["SEQUENCE"] != subject]
                                
print(raw_data_set.shape)

"""
Benzerliği %30'un altında olan proteinler kaldıktan sonra, her bir sınıfta
kaç protein kaldığı kontrol edilir. Eğer sınıflar arasındaki protein sayıları
birbirinden çok farklıysa veri seti dengeli hale getirilmelidir.
"""
print("DBP_size:", raw_data_set[raw_data_set["TYPE"] == "dbp"].shape)
print("RBP_size:", raw_data_set[raw_data_set["TYPE"] == "rbp"].shape)
print("NNABP_size:", raw_data_set[raw_data_set["TYPE"] == "nnabp"].shape)
print("Dataset_size:", raw_data_set.shape)

"""
Benzerlikler hesaplandıktan sonra veri setinde, 3602 DBP, 1769 RBP ve
6933 NNABP kalmıştır. Veri setinin dengeli hale getirilmesi için en az
sınıf olan RBP'ye göre rastgele veriler çıkartılacaktır. Bununla birlikte,
sample metotu kullanılarak her sınıftan 1500'er olmak üzere veri setinde
toplamda 4500 protein verisi kalacaktır.
"""
dbp_after_blast_data = raw_data_set[raw_data_set["TYPE"] == "dbp"].sample(n = 1500)
rbp_after_blast_data = raw_data_set[raw_data_set["TYPE"] == "rbp"].sample(n = 1500)
nnabp_after_blast_data = raw_data_set[raw_data_set["TYPE"] == "nnabp"].sample(n = 1500)

raw_data_set = pd.concat([dbp_after_blast_data, rbp_after_blast_data, nnabp_after_blast_data]).reset_index(drop = True)
print("Dataset_size:", raw_data_set.shape)

"""
Temizlenmiş veri seti, 'cleaned_data_set' isimli yeni bir değişkene atanır.
"""
cleaned_data_set = raw_data_set.reset_index(drop = True)

"""
DBP'lerin, RBP'lerin, NNABP'lerin ve
Temizlenen nihai veri setinin satır ve sütun boyutları kontrol edilir.
"""
print("DBP_size:", cleaned_data_set[cleaned_data_set["TYPE"] == "dbp"].shape)
print("RBP_size:", cleaned_data_set[cleaned_data_set["TYPE"] == "rbp"].shape)
print("NNABP_size:", cleaned_data_set[cleaned_data_set["TYPE"] == "nnabp"].shape)
print("Dataset_size:", cleaned_data_set.shape)

"""
Veri seti csv dosyası olarak yazılır.
"""
cleaned_data_set.to_csv("../dataset/main_data_set(cleaned).csv")
