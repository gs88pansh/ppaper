

rem 数据预处理 diginetica
rem python preprocess.py diginetica ./diginetica/data/raw/train-item-views.csv 5 15 15 3 7

rem 数据预处理 yoochoose
rem python preprocess.py yoochoose ./yoochoose/data/raw/yoochoose-training.txt 5 15 15 3 1

rem 训练数据 diginetica i2v
python train_i2v.py 256 0.0007 30 100 512 39187 ./diginetica/data/preprocessed/i2v.txt ./diginetica/model

rem 训练数据 diginetica ri2v
python train_ri2v.py diginetica 2048 0.0002 40 100 256 100  em100ba512sa256le0.0007 A30 0 0.8 39187
python train_ri2v.py diginetica 4096 0.0002 40 100 256 100  em100ba512sa256le0.0007 A30 0 0.8 39187
python train_ri2v.py diginetica 2048 0.0002 40 100 256 200  em100ba512sa256le0.0007 A30 0 0.8 39187
python train_ri2v.py diginetica 4096 0.0002 40 100 256 200  em100ba512sa256le0.0007 A30 0 0.8 39187
python train_ri2v.py diginetica 2048 0.0004 40 100 256 100  em100ba512sa256le0.0007 A30 0 0.8 39187

