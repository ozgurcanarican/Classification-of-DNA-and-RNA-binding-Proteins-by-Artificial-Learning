# EĞİTİLMİŞ MODELLERİN PYTHON'A YÜKLENMESİ

import pickle

rf1 = pickle.load(open("../pipeline_1/pipeline1_rf(initial)", "rb"))
svm1 = pickle.load(open("../pipeline_1/pipeline1_svm(initial)", "rb"))
cnn1 = pickle.load(open("../pipeline_1/pipeline1_cnnbilstm(initial)", "rb"))
mlp1 = pickle.load(open("../pipeline_1/pipeline1_mlp(initial)", "rb"))

rf2 = pickle.load(open("../pipeline_2/pipeline2_rf(selected_6000)", "rb"))
svm2 = pickle.load(open("../pipeline_2/pipeline2_svm(selected_6000)", "rb"))
cnn2 = pickle.load(open("../pipeline_2/pipeline2_cnnbilstm(selected_6000)", "rb"))
mlp2 = pickle.load(open("../pipeline_2/pipeline2_mlp(selected_6000)", "rb"))

rf3 = pickle.load(open("../pipeline_3/pipeline3_rf(selected_4000)", "rb"))
svm3 = pickle.load(open("../pipeline_3/pipeline3_svm(selected_4000)", "rb"))
cnn3 = pickle.load(open("../pipeline_3/pipeline3_cnnbilstm(selected_4000)", "rb"))
mlp3 = pickle.load(open("../pipeline_3/pipeline3_mlp(selected_4000)", "rb"))

rf4 = pickle.load(open("../pipeline_4/pipeline4_rf(selected_2000)", "rb"))
svm4 = pickle.load(open("../pipeline_4/pipeline4_svm(selected_2000)", "rb"))
cnn4 = pickle.load(open("../pipeline_4/pipeline4_cnnbilstm(selected_2000)", "rb"))
mlp4 = pickle.load(open("../pipeline_4/pipeline4_mlp(selected_2000)", "rb"))

rf5 = pickle.load(open("../pipeline_5/pipeline5_rf(selected_1000)", "rb"))
svm5 = pickle.load(open("../pipeline_5/pipeline5_svm(selected_1000)", "rb"))
cnn5 = pickle.load(open("../pipeline_5/pipeline5_cnnbilstm(selected_1000)", "rb"))
mlp5 = pickle.load(open("../pipeline_5/pipeline5_mlp(selected_1000)", "rb"))

rf6 = pickle.load(open("../pipeline_6/pipeline6_rf(selected_500)", "rb"))
svm6 = pickle.load(open("../pipeline_6/pipeline6_svm(selected_500)", "rb"))
cnn6 = pickle.load(open("../pipeline_6/pipeline6_cnnbilstm(selected_500)", "rb"))
mlp6 = pickle.load(open("../pipeline_6/pipeline6_mlp(selected_500)", "rb"))


#*****************************************************************************
# TEST474 VE PDB255 BAĞIMSIZ VERİ SETLERİNİN YÜKLENMESİ

import pandas as pd
import numpy as np

test474_1 = pd.read_csv(r"../alternative_dataset/test474(extracted).csv", index_col = [0])
pdb255_1 = pd.read_csv(r"../alternative_dataset/pdb255(extracted).csv", index_col = [0])

test474_2 = pd.read_csv(r"../alternative_dataset/test474(selected_6000).csv", index_col = [0])
pdb255_2 = pd.read_csv(r"../alternative_dataset/pdb255(selected_6000).csv", index_col = [0])

test474_3 = pd.read_csv(r"../alternative_dataset/test474(selected_4000).csv", index_col = [0])
pdb255_3 = pd.read_csv(r"../alternative_dataset/pdb255(selected_4000).csv", index_col = [0])

test474_4 = pd.read_csv(r"../alternative_dataset/test474(selected_2000).csv", index_col = [0])
pdb255_4 = pd.read_csv(r"../alternative_dataset/pdb255(selected_2000).csv", index_col = [0])

test474_5 = pd.read_csv(r"../alternative_dataset/test474(selected_1000).csv", index_col = [0])
pdb255_5 = pd.read_csv(r"../alternative_dataset/pdb255(selected_1000).csv", index_col = [0])

test474_6 = pd.read_csv(r"../alternative_dataset/test474(selected_500).csv", index_col = [0])
pdb255_6 = pd.read_csv(r"../alternative_dataset/pdb255(selected_500).csv", index_col = [0])


#*****************************************************************************
# BAĞIMSIZ VERİ SETLERİYLE MODELLERE TAHMİNLEME YAPTIRILMASI

IN_PROCESS_CLF = rf1
IN_PROCESS_DF = test474_1

X = IN_PROCESS_DF.drop(["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"], axis = 1).to_numpy()
y = IN_PROCESS_DF["TYPE"]

#X = X.reshape(X.shape[0], X.shape[1], 1) #RF ve SVM de yok
        
import timeit
from sklearn.preprocessing import OneHotEncoder
        
categorical_encoding = {"nnabp": 0, "dbp": 1, "rbp": 2}
integer_encoded = np.array(y.replace(categorical_encoding))
        
one_hot_encoder = OneHotEncoder(sparse = False)
y = one_hot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))
y = np.argmax(y, axis = 1) #CNN-BiLSTM ve MLP de yok
        
from sklearn.metrics import classification_report

time_start = timeit.default_timer()
y_pred = IN_PROCESS_CLF.predict(X) # CNN-BiLSTM ve MLP de "predict_classes", diğerleri "predict"
time_stop = timeit.default_timer()

training_time = (time_stop - time_start)

#y_pred = one_hot_encoder.fit_transform(y_pred.reshape(len(y_pred), 1)) #RF ve SVM de yok

#*****************************************************************************
# SONUÇLARIN KAYDEDİLMESİ

IN_PROCESS_FNAME = "pipeline1_rf_test474(initial).txt"
IN_PROCESS_HEADER = "PIPELINE-1 RF RESULTS ON TEST474"
IN_PROCESS_CONDITION_ONE = "    -Feature selection not applied"
IN_PROCESS_CONDITION_TWO = "    -Random Forest Classifier trained"

results = open(IN_PROCESS_FNAME, "w")
results.write(IN_PROCESS_HEADER)
results.write("\n\n" + IN_PROCESS_CONDITION_ONE)
results.write("\n" + IN_PROCESS_CONDITION_TWO)
results.write("\n\nPerformance Score:")
results.write("\n\n" + classification_report(y, y_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))
results.write("\n\nRuntime: " + str(training_time) + " seconds")
results.close()


#*****************************************************************************
# ROC HESAPLANMASI

from sklearn.metrics import roc_curve, auc

y = one_hot_encoder.fit_transform(y.reshape(len(y), 1)) #CNN-BiLSTM ve MLP'de yok
y_score = np.array(IN_PROCESS_CLF.predict_proba(X))
n_classes = y.shape[1]

false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()

for i in range(n_classes):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y[:, i], y_score[:, i])
    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

all_false_positive_rates = np.unique(np.concatenate([false_positive_rate[i] for i in range(n_classes)]))
mean_true_positive_rates = np.zeros_like(all_false_positive_rates)

for i in range(n_classes):
    mean_true_positive_rates += np.interp(all_false_positive_rates, false_positive_rate[i], true_positive_rate[i])
    
mean_true_positive_rates /= n_classes

false_positive_rate["macro"] = all_false_positive_rates
true_positive_rate["macro"] = mean_true_positive_rates
roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])


#*****************************************************************************
# ROC AUC GRAFİK ÇİZDİRME VE KAYDETME

import matplotlib.pyplot as plt
from itertools import cycle

plt.figure()

plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
         label = "micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
         color = "gold", linestyle = ":", linewidth = 4)

plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
         label = "macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
         color = "darkmagenta", linestyle = ":", linewidth = 4)

colors = cycle(["firebrick", "forestgreen", "deepskyblue"])
classes = {0: "NNABP", 1: "DBP", 2: "RBP"}
for i, color in zip(range(n_classes), colors):
    plt.plot(false_positive_rate[i], true_positive_rate[i], color = color, lw = 2,
             label = "ROC curve of {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], "k--", lw = 2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Random Forest Classifier (Max Data Complexity)")
plt.legend(loc = "lower right")

roc = plt.gcf()
plt.show()

roc.savefig("pipeline1_rf_test474(initial)_ROC")
