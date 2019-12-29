HRNet ==> https://arxiv.org/pdf/1902.09212.pdf (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)

(1) Finetune the HRNet-64 with the training data (input size: 224x224). The validation accuracy is expected to achieve 70%. 
    -- train_iNat2019.py (Note that the input size should be 224x224).

(2) Then, finetune the above model by feeding the training data input size of 448x448.The validation accuracy is expected to acieve 78%.
    -- train_iNat2019.py (Note that the input size should be 448x448).

(3) Next, finetune the above model with class-wise balanced sampling. The validation accuracy is expected to achieve 80%+.
    -- The balanced sampling implementation in included in train_iNat2019.py

(4) Continue to finetune the above model with a few epochs by including the CutOut data augmentation. When submitting to the testing predictions,
    the Error Rate is expected to be around 19%.
    -- The CutOut implementation is included in data_loader.py

(5) Finetune the above model with the validation data (3000 images). The Error Rate is expected to be around 18%.
    -- train_iNat2019.py data_loader.py

(6) Voting all the trained models with the ensembling code (provided by Kaggle).

