# VERİ SETLERİNİN PYTHON'A YÜKLENMESİ

import pandas as pd
import timeit

initial_dataset = pd.read_csv(r"../dataset/main_data_set(extracted).csv", index_col = [0])
six_k_dataset = pd.read_csv(r"../dataset/main_data_set(selected_6000).csv", index_col = [0])
four_k_dataset = pd.read_csv(r"../dataset/main_data_set(selected_4000).csv", index_col = [0])
two_k_dataset = pd.read_csv(r"../dataset/main_data_set(selected_2000).csv", index_col = [0])
one_k_dataset = pd.read_csv(r"../dataset/main_data_set(selected_1000).csv", index_col = [0])
five_h_dataset = pd.read_csv(r"../dataset/main_data_set(selected_500).csv", index_col = [0])

print(initial_dataset.shape)
print(six_k_dataset.shape)
print(four_k_dataset.shape)
print(two_k_dataset.shape)
print(one_k_dataset.shape)
print(five_h_dataset.shape)


#**********************************************************************************************
# CNN-BILSTM MODELİNİN OLUŞTURULMASI

import numpy as np

IN_PROCESS_DATASET = initial_dataset

X = IN_PROCESS_DATASET.drop(["UNIPROT_ID", "TYPE", "SEQUENCE", "LENGTH"], axis = 1).to_numpy()
y = IN_PROCESS_DATASET["TYPE"]

from sklearn.preprocessing import OneHotEncoder

categorical_encoding = {"nnabp": 0, "dbp": 1, "rbp": 2}
integer_encoded = np.array(IN_PROCESS_DATASET["TYPE"].replace(categorical_encoding))

one_hot_encoder = OneHotEncoder(sparse = False)
y = one_hot_encoder.fit_transform(integer_encoded.reshape(len(integer_encoded), 1))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

initial_dimension = 8431
six_k_dimension = 6000
four_k_dimension = 4000
two_k_dimension = 2000
one_k_dimension = 1000
five_h_dimension = 500

FEATURE_SIZE = initial_dimension

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Bidirectional, LSTM, Dropout

time_start = timeit.default_timer()

model = Sequential()
model.add(Conv1D(64, 3, activation = "relu", input_shape = (FEATURE_SIZE, 1)))
model.add(MaxPooling1D(pool_size = 2, strides = 1, padding = "same"))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(3, activation = 'softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])

model.fit(X_train, y_train, batch_size = 16, epochs = 100, verbose = 0)

time_stop = timeit.default_timer()


#**********************************************************************************************
# BAŞARI ÖLÇÜTLERİNİN HESAPLANMASI

from sklearn.metrics import classification_report

prd = model.predict_classes(X_train)
y_train_pred = one_hot_encoder.fit_transform(prd.reshape(len(prd), 1))
print(classification_report(y_train, y_train_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))

prd_tst = model.predict_classes(X_test)
y_pred = one_hot_encoder.fit_transform(prd_tst.reshape(len(prd_tst), 1))
print(classification_report(y_test, y_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))

training_time = (time_stop - time_start)
print("Runtime:", training_time)


#**********************************************************************************************
# SONUÇLARIN KAYDEDİLMESİ

initial_results_fname = "pipeline1_cnnbilstm_results(initial).txt"
six_k_results_fname = "pipeline2_cnnbilstm_results(selected_6000).txt"
four_k_results_fname = "pipeline3_cnnbilstm_results(selected_4000).txt"
two_k_results_fname = "pipeline4_cnnbilstm_results(selected_2000).txt"
one_k_results_fname = "pipeline5_cnnbilstm_results(selected_1000).txt"
five_h_results_fname = "pipeline6_cnnbilstm_results(selected_500).txt"

initial_results_header = "PIPELINE-1 CNN-BILSTM RESULTS"
six_k_results_header = "PIPELINE-2 CNN-BILSTM RESULTS"
four_k_results_header = "PIPELINE-3 CNN-BILSTM RESULTS"
two_k_results_header = "PIPELINE-4 CNN-BILSTM RESULTS"
one_k_results_header = "PIPELINE-5 CNN-BILSTM RESULTS"
five_h_results_header = "PIPELINE-6 CNN-BILSTM RESULTS"

initial_condition = "    -Feature selection not applied"
six_k_condition = "    -6000 features selected"
four_k_condition = "    -4000 features selected"
two_k_condition = "    -2000 features selected"
one_k_condition = "    -1000 features selected"
five_h_condition = "    -500 features selected"

IN_PROCESS_FNAME = initial_results_fname
IN_PROCESS_HEADER = initial_results_header
IN_PROCESS_CONDITION_ONE = initial_condition
IN_PROCESS_CONDITION_TWO = "    -CNN-BiLSTM Classifier trained"

results = open(IN_PROCESS_FNAME, "w")
results.write(IN_PROCESS_HEADER)
results.write("\n\n" + IN_PROCESS_CONDITION_ONE)
results.write("\n" + IN_PROCESS_CONDITION_TWO)
results.write("\n\nTrain Score:")
results.write("\n\n" + classification_report(y_train, y_train_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))
results.write("\n\nTest Score:")
results.write("\n\n" + classification_report(y_test, y_pred, target_names = ["nnabp", "dbp", "rbp"], digits = 3))
results.write("\n\nRuntime: " + str(training_time) + " seconds")
results.close()


#**********************************************************************************************
# ROC HESAPLANMASI

from sklearn.metrics import roc_curve, auc

y_score = model.predict_proba(X_test)
n_classes = y.shape[1]

false_positive_rate = dict()
true_positive_rate = dict()
roc_auc = dict()

for i in range(n_classes):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

all_false_positive_rates = np.unique(np.concatenate([false_positive_rate[i] for i in range(n_classes)]))
mean_true_positive_rates = np.zeros_like(all_false_positive_rates)

for i in range(n_classes):
    mean_true_positive_rates += np.interp(all_false_positive_rates, false_positive_rate[i], true_positive_rate[i])
    
mean_true_positive_rates /= n_classes

false_positive_rate["macro"] = all_false_positive_rates
true_positive_rate["macro"] = mean_true_positive_rates
roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])


#**********************************************************************************************
# ROC AUC GRAFİK ÇİZDİRME

initial_title = "ROC Curves for CNN-BiLSTM Classifier (Max Data Complexity)"
six_k_title = "ROC Curves for CNN-BiLSTM Classifier (6000 Features Selected)"
four_k_title = "ROC Curves for CNN-BiLSTM Classifier (4000 Features Selected)"
two_k_title = "ROC Curves for CNN-BiLSTM Classifier (2000 Features Selected)"
one_k_title = "ROC Curves for CNN-BiLSTM Classifier (500 Features Selected)"
five_h_title = "ROC Curves for CNN-BiLSTM Classifier (Min Data Complexity)"

IN_PROCESS_ROC_TITLE = initial_title

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
plt.title(IN_PROCESS_ROC_TITLE)
plt.legend(loc = "lower right")

roc = plt.gcf()
plt.show()


#**********************************************************************************************
# GRAFİĞİN KAYDEDİLMESİ

initial_roc_fname = "pipeline1_cnnbilstm_ROC(initial).png"
six_k_roc_fname = "pipeline2_cnnbilstm_ROC(selected_6000).png"
four_k_roc_fname = "pipeline3_cnnbilstm_ROC(selected_4000).png"
two_k_roc_fname = "pipeline4_cnnbilstm_ROC(selected_2000).png"
one_k_roc_fname = "pipeline5_cnnbilstm_ROC(selected_1000).png"
five_h_roc_fname = "pipeline6_cnnbilstm_ROC(selected_500).png"

IN_PROCESS_ROC_FNAME = initial_roc_fname

roc.savefig(IN_PROCESS_ROC_FNAME)


#**********************************************************************************************
# MODELİN KAYDEDİLMESİ

initial_model = "pipeline1_cnnbilstm(initial)"
six_k_model = "pipeline2_cnnbilstm(selected_6000)"
four_k_model = "pipeline3_cnnbilstm(selected_4000)"
two_k_model = "pipeline4_cnnbilstm(selected_2000)"
one_k_model = "pipeline5_cnnbilstm(selected_1000)"
five_h_model = "pipeline6_cnnbilstm(selected_500)"

IN_PROCESS_MODEL = initial_model

import pickle

pickle.dump(model, open(IN_PROCESS_MODEL, "wb"))