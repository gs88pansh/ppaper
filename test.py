import sys
sys.path.append("./")
from data_ri2v import *

args = sys.argv
print("后面的依次为参数："
      "\n \t data_set_name", args[1],
      "\n \t test_model", args[2],
      "\n \t embedding_size", args[3],
      "\n \t batch_size", args[4],
      "\n \t hidden_size", args[5],
      "\n \t model_dir", args[6],
      "\n \t restore_model", args[7],
      "\n \t item_size", args[8],
      ),

#embedding_size, rnn_states, item_size, batch_size, n_sampled, learning_rate, joint_train = \
#    100, 100, 43136, 128, 2000, 0.0001, False


class Args(object):
    dataSetName = args[1]
    this_base_dir = "./{}".format(dataSetName)
    test_model = args[2]
    embedding_size = int(args[3])
    batch_size = int(args[4])
    hidden_size = int(args[5])
    restore_dir = this_base_dir + "/model/{}/".format(test_model) + args[6]
    restore_model = args[7]
    item_size = int(args[8])


    joint_train = True
    keep_prob = 1

    n_sampled = 0
    learning_rate = 0
    epochs = 0
    test_seq_path = this_base_dir + "/data/preprocessed/test_seq.txt"

    saver_dir = ""
    summary_path = ""

    D = RI2VDataSet()
    log_dir = this_base_dir + "/model/TEST.txt"

if __name__ == "__main__":
    model_name = "ri2v"
    train = True
    args = Args()
    testRnnProcess(args, model_name)
