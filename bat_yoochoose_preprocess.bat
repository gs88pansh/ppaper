rem dataSetName         yoochoose
rem raw_data_file,      ./yoochoose/data/row/yoochoose-training.txt
rem item_num_filter,    5
rem session_num_max,    15
rem num_last_days,      15
rem window_size,        3
rem test_days,          1
rem train_seq_path,     ./yoochoose/data/preprocessed/train-seq.txt
rem test_seq_path,      ./yoochoose/data/preprocessed/test-seq.txt
rem last_n_seq_path,    ./yoochoose/data/preprocessed/last-15-seq.txt
rem i2v_path            ./yoochoose/data/preprocessed/i2v.txt

python preprocess.py ^
yoochoose ^
./yoochoose/data/raw/yoochoose-training.txt ^
5 ^
15 ^
15 ^
3 ^
1
