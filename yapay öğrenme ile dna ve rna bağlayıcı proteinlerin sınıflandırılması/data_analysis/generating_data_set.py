"""
Verileri okumak için pandas kütüphanesi çağrılır
"""
import pandas as pd

"""
DNA-bağlayıcı, RNA-bağlayıcı ve nükleik asitlerin bağlanmadığı proteinlerin ayrı ayrı
bulunduğu üç veriseti pandas ile çağrılır. İlgili verisetleri, sınıf bakımından kendilerini
tanımlayan 'dbp_raw_data', 'rbp_raw_data' ve 'nnabp_raw_data' isimli değişkenlere
atanır.
"""
dbp_raw_data = pd.read_excel(r"../database_ids/DNA_BINDING.xlsx")
rbp_raw_data = pd.read_excel(r"../database_ids/RNA_BINDING.xlsx")
nnabp_raw_data = pd.read_excel(r"../database_ids/NON_NUCLEIC_ACID_BINDING.xlsx")

"""
Veri setleri içerisinde, proteinlerin Uniprot kimlikleri, uzunlukları ve dizi 
bilgileri bulunmaktadır. Bu bilgileri içeren sütunların, sütun isimleri düzenlenir.
"""
dbp_raw_data.rename(columns = {'Entry':'UNIPROT_ID', 'Length':'LENGTH', 'Sequence':'SEQUENCE'}, inplace = True)
rbp_raw_data.rename(columns = {'Entry':'UNIPROT_ID', 'Length':'LENGTH', 'Sequence':'SEQUENCE'}, inplace = True)
nnabp_raw_data. rename(columns = {'Entry':'UNIPROT_ID', 'Length':'LENGTH', 'Sequence':'SEQUENCE'}, inplace = True)

"""
Üç ayrı sınıfın proteinleri henüz birleştirilmemişken, protein sınıflarının
etiketlendiği yeni birer sütun oluşturulur.
'TYPE' isimli birer sütun ismi verilir ve;
    DNA-bağlayıcı proteinler ----> dbp
    RNA-bağlayıcı proteinler ----> rbp
    Nükleotitlere bağlanmayan proteinler ----> nnabp
olarak etiketlenir.
"""
dbp_raw_data["TYPE"] = "dbp"
rbp_raw_data["TYPE"] = "rbp"
nnabp_raw_data["TYPE"] = "nnabp"

"""
Üç ayrı sınıfın bulunduğu üç veri seti, 'UNIPROT_ID' sütunundan birleştirilerek
tek bir veri seti haline getirilir.
"""
main_data_set = pd.concat([dbp_raw_data, rbp_raw_data, nnabp_raw_data]).reset_index(drop = True)

"""
DBP'lerin, RBP'lerin, NNABP'lerin ve
Oluşturulan nihai veri setinin satır ve sütun boyutları kontrol edilir.
"""
print("DBP_size:", main_data_set[main_data_set["TYPE"] == "dbp"].shape)
print("RBP_size:", main_data_set[main_data_set["TYPE"] == "rbp"].shape)
print("NNABP_size:", main_data_set[main_data_set["TYPE"] == "nnabp"].shape)
print("Dataset_size:", main_data_set.shape)

"""
Veri seti csv dosyası olarak yazılır.
"""
main_data_set.to_csv(r"../dataset/main_data_set(raw).csv")
