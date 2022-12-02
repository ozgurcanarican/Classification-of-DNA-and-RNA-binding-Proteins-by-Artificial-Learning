import pandas as pd

original_test474 = pd.read_csv(r"test474(extracted).csv", index_col = [0])
original_pdb255 = pd.read_csv(r"pdb255(extracted).csv", index_col = [0])

deepdrbp2l_test474 = pd.read_excel(r"deepdrbp2l_test474_results.xlsx", index_col = [0])
deepdrbp2l_pdb255 = pd.read_excel(r"deepdrbp2l_pdb255_results.xlsx", index_col = [0])

idrbpmmc_test474 = pd.read_excel(r"idrbpmmc_test474_results.xlsx", index_col = [0])
idrbpmmc_pdb255 = pd.read_excel(r"idrbpmmc_pdb255_results.xlsx", index_col = [0])


ORIGINAL_DATA = original_pdb255
IN_PROCESS_MODEL = idrbpmmc_pdb255

y = ORIGINAL_DATA["TYPE"]
y_pred = IN_PROCESS_MODEL["TYPE"]

from sklearn.metrics import classification_report


#*****************************************************************************
# SONUÇLARIN KAYDEDİLMESİ

IN_PROCESS_FNAME = "idrbpmmc_pdb255_results.txt"
IN_PROCESS_HEADER = "iDRBP_MMC RESULTS ON PDB255"

results = open(IN_PROCESS_FNAME, "w")
results.write(IN_PROCESS_HEADER)
results.write("\n\nPerformance Score:")
results.write("\n\n" + classification_report(y, y_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))
results.close()


#*****************************************************************************
# HATA MATRİSİ OLUŞTURULMASI

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

hata_matrisi = confusion_matrix(y, y_pred)
heatmap = sns.heatmap(hata_matrisi, annot = True, cmap = "Greens", fmt = "g",
            xticklabels = {"NNABP": 0, "DBP": 1, "RBP": 2}, yticklabels = {"NNABP": 0, "DBP": 1, "RBP": 2})
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation = 0)
heatmap.axhline(y = 0, color = 'k', linewidth = 4)
heatmap.axhline(y = 3, color = 'k', linewidth = 4)
heatmap.axvline(x = 0, color = 'k', linewidth = 4)
heatmap.axvline(x = 3, color = 'k', linewidth = 4)

plt.title("iDRBP_MMC Confusion Matrix (PDB255)", fontsize = 15)
plt.xlabel("Predicted", fontsize = 12)
plt.ylabel("Actual", fontsize = 12)

plt.tight_layout()
hm = plt.gcf()
plt.show()

hm.savefig("idrbpmmc_pdb255_cmap")