import sys
sys.path.append("./")
from data_ri2v import *

args = sys.argv




print("后面的依次为参数："

      "\n \t data_set_name", args[1],
      "\n \t n_sampled", args[2],
      "\n \t learning_rate", args[3],
      "\n \t epochs", args[4],
      "\n \t embedding_size", args[5],
      "\n \t batch_size", args[6],
      "\n \t hidden_size", args[7],
      "\n \t i2v_model_dir", args[8],
      "\n \t restore_model", args[9],
      "\n \t joint_train", args[10],
      "\n \t keep_drop", args[11],
      "\n \t item_size", args[12],

      ),

#embedding_size, rnn_states, item_size, batch_size, n_sampled, learning_rate, joint_train = \
#    100, 100, 43136, 128, 2000, 0.0001, False


class Args(object):
    dataSetName = args[1]
    this_base_dir = "./{}".format(dataSetName)

    n_sampled = int(args[2])
    learning_rate = float(args[3])
    epochs = int(args[4])
    embedding_size = int(args[5])
    batch_size = int(args[6])
    hidden_size = int(args[7])

    restore_i2v_dir = this_base_dir + "/model/i2v/" + args[8] # dataSetName
    restore_i2v_model = args[9]

    joint_train = int(args[10]) == 1
    keep_prob = float(args[11])
    item_size = int(args[12])
    is_random_added = int(args[13]) == 1 # 输入数据加随机

    training_seq_path = this_base_dir + "/data/preprocessed/train-seq.txt"
    testing_seq_path = this_base_dir + "/data/preprocessed/test-seq.txt"

    restore_ri2v_dir = ""
    restore_ri2v_model = ""
    restore_dir = ""
    restore_model = ""

    saver_dir = this_base_dir + "/model/ri2v" \
        + "/em{}hi{}ba{}sa{}le{:6f}dr{:.2f}random{}"\
            .format(embedding_size, hidden_size, batch_size, n_sampled, learning_rate, 1-keep_prob, is_random_added)

    summary_path = this_base_dir + "/model/ri2v" \
        + "/view_em{}hi{}ba{}sa{}le{:6f}dr{:.2f}random{}"\
            .format(embedding_size, hidden_size, batch_size, n_sampled, learning_rate, 1-keep_prob, is_random_added)

    D = RI2VDataSet()
    log_dir = this_base_dir + "/model/README.txt"

if __name__ == "__main__":
    model_name = "ri2v"
    train = True
    args = Args()

    trainRnnProcess(args, model_name)
#n_sample,rnn_states,embedding_size,joint_train
