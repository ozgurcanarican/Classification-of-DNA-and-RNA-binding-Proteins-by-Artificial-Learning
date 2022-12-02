import pandas as pd
import requests

links = ["http://bliulab.net/iDRBP_MMC/static/dataset/test_dataset_TEST474.txt",
         "http://bliulab.net/iDRBP_MMC/static/dataset/test_dataset_PDB255.txt"]

url_one = requests.get(links[0])
url_two = requests.get(links[1])

test474 = url_one.text
test474 = test474.splitlines()[6:]

pdb255 = url_two.text
pdb255 = pdb255.splitlines()[5:]

dataset = [test474, pdb255]
dataset_name = ["test474", "pdb255"]


#******************************************************************************
# BAĞIMSIZ VERİSETLERİNİ ELDE ETME

for d in range(len(dataset)):

    dbpHandler, rbpHandler, nnabpHandler, otherHandler = False, False, False, False
    ids, seq, types = [], [], []
    
    for line in dataset[d][:]:
        
        dataset[d].remove(line)
        
        if line == "==============(1) Proteins that bind to DNA ================":
            dbpHandler = True
            rbpHandler = False
            nnabpHandler = False
            otherHandler = False
            
        elif line == "==============(2) Proteins that bind to RNA ================":
            dbpHandler = False
            rbpHandler = True
            nnabpHandler = False
            otherHandler = False
            
        elif line == "=========(3) Proteins that bind to both DNA and RNA=========":
            dbpHandler = False
            rbpHandler = False
            nnabpHandler = False
            otherHandler = True
            
        elif line == "=======(3) Proteins that neither bind to DNA nor RNA========":
            dbpHandler = False
            rbpHandler = False
            nnabpHandler = True
            otherHandler = False
            
        elif line == "========(4) Proteins that neither bind to DNA nor RNA=======":
            dbpHandler = False
            rbpHandler = False
            nnabpHandler = True
            otherHandler = False
            
        if line.startswith(">"):
            
            if dbpHandler:
                types.append("dbp")
            elif rbpHandler:
                types.append("rbp")
            elif nnabpHandler:
                types.append("nnabp")
            elif otherHandler:
                types.append("other")
                
            ids.append(line)
            seq.append("")
            for nline in dataset[d]:
                if nline.startswith(">"):
                    break
                if not (nline.startswith("-") or nline.startswith("=")):
                    seq[-1] = seq[-1] + nline
                    
                    
    seq_len = []
    for s in seq:
        seq_len.append(len(s))
    
    
    alternative_dataset = pd.DataFrame({"DATABASE_ID" : ids, "TYPE" : types,
                                        "SEQUENCE" : seq, "LENGTH" : seq_len})
               
    print(alternative_dataset.shape)
    
    alternative_dataset = alternative_dataset[alternative_dataset["TYPE"] != "other"]
    print(alternative_dataset.shape)
    
    aaList = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    for seq in alternative_dataset["SEQUENCE"]:
        seqIndex = 0
        isSequenceCorrect = True
        while seqIndex <= (len(seq) - 1) and isSequenceCorrect:
            if not seq[seqIndex] in aaList:
                alternative_dataset = alternative_dataset[alternative_dataset["SEQUENCE"] != seq]
                isSequenceCorrect = False
            seqIndex += 1
    print(alternative_dataset.shape)
    
    alternative_dataset.drop_duplicates(subset = ["SEQUENCE"], inplace = True)
    print(alternative_dataset.shape)
    
    alternative_dataset = alternative_dataset.reset_index(drop = True)
    
    alternative_dataset.to_csv("../alternative_dataset/{}(main).csv".format(dataset_name[d]))