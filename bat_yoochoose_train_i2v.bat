


rem n_sampled
rem learning_rate
rem epoches
rem embedding_size
rem batch_size
rem item_size
rem train_file
rem model_base_dir


python train_i2v 256 0.001 40 100 512 ${item_size} ./yoochoose/data/preprocessed/i2v.txt ./yoochoose/model