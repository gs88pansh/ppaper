


rem n_sampled
rem learning_rate
rem epoches
rem embedding_size
rem batch_size
rem item_size
rem train_file
rem model_base_dir

rem python train_i2v.py ${sa} ${le} ${epoches} ${em} ${ba} ${item_size} ${train_file} ${model_base_dir}
python train_i2v.py 256 0.0007 30 100 512 39187 ./diginetica/data/preprocessed/i2v.txt ./diginetica/model