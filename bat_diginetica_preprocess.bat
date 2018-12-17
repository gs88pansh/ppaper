rem dataSetName         diginetica
rem raw_data_file,      ./diginetica/data/row/train-item-views.csv
rem item_num_filter,    5
rem session_num_max,    15
rem num_last_days,      15
rem window_size,        3
rem test_days,          1
rem train_seq_path,     ./diginetica/data/preprocessed/train-seq.txt
rem test_seq_path,      ./diginetica/data/preprocessed/test-seq.txt
rem last_n_seq_path,    ./diginetica/data/preprocessed/last-15-seq.txt
rem i2v_path            ./diginetica/data/preprocessed/i2v.txt

python preprocess.py ^
diginetica ^
./diginetica/data/raw/train-item-views.csv ^
5 ^
15 ^
15 ^
3 ^
7