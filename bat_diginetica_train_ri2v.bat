
rem data_set_name, args[1],
rem n_sampled, args[2],
rem learning_rate, args[3],
rem epochs, args[4],
rem embedding_size, args[5],
rem batch_size, args[6],
rem hidden_size, args[7],
rem i2v_model_dir, args[8],
rem restore_model, args[9],
rem joint_train, args[10],
rem keep_drop, args[11]
rem item_size, args[12]

rem python train_ri2v.py diginetica ${sa} ${le} ${e} ${em} ${ba} ${hi} ${i2v_model_dir} ${restore_model} 0 ${dr} ${item_size}


python train_ri2v.py diginetica 2048 0.0002 40 100 256 100  em100ba512sa256le0.0007 A1 0 0.8 39187
python train_ri2v.py diginetica 4096 0.0002 40 100 256 100  em100ba512sa256le0.0007 A1 0 0.8 39187
python train_ri2v.py diginetica 2048 0.0002 40 100 256 200  em100ba512sa256le0.0007 A1 0 0.8 39187
python train_ri2v.py diginetica 4096 0.0002 40 100 256 200  em100ba512sa256le0.0007 A1 0 0.8 39187
python train_ri2v.py diginetica 2048 0.0002 40 100 256 100  em100ba512sa256le0.0007 A1 0 0.5 39187