
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

rem python train_reri2v.py diginetica 2048 0.0001 30 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.2 39187
rem python train_reri2v.py diginetica 2048 0.0001 30 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.4 39187
rem python train_reri2v.py diginetica 2048 0.0001 30 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.6 39187
rem python train_reri2v.py diginetica 2048 0.0001 30 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.8 39187

rem python train_reri2v.py diginetica 2048 0.00008 50 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.5 39187
python train_reri2v.py diginetica 2048 0.00004 100 100 64 200 em100hi200ba256sa2048le0.001000dr0.20 A30 1 0.5 39187