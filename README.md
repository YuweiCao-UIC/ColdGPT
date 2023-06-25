## ColdGPT
This repository contains the source code of our paper **Multi-task Item-attribute Graph Pre-training for Strict Cold-start Item Recommendation** (submitted to RecSys 2023).

To use ColdGPT:
1) download data folder from:
https://drive.google.com/drive/folders/1Vdu9N0p8bOW9B5cA3GgN3M5SolVY3Img?usp=sharing Place the downloaded data folder inside this folder.

2) run ColdGPT.py to pretrain a bipartite item-attribute graph. E.g.:
```
python ColdGPT.py --t1 --t3 --plm SBERT
```

3) run evaluate.py to insert the SCS items into the pretained item-attribute graph. Extract the embeddings of the items for making racommendations. E.g.:
```
python evaluate.py --t1 --t3 --plm SBERT
```

Links to the Google drive folders containing all four preprocessed SCS datasets will be uploaded soon.
