import sys
sys.path.append("./")
from _shallow_model_iknn import *
import _metric_accurate as metric
from Utils import evaluate_sessions, log

log_dir = "./_SHALLOW_MODEL.txt"
args = sys.argv
dataSetName = args[1]


log(log_dir, "-----model_name:{} data_set_name:{}".format("iknn",dataSetName))

print("data_set_name:", args[1])


iknn = ItemKNN()

# load data
train_expand = pd.read_csv("./{}/data/preprocessed/train-expand.txt".format(dataSetName), sep='\t', dtype={'ItemId':np.int64})
test_expand = pd.read_csv("./{}/data/preprocessed/test-expand.txt".format(dataSetName), sep='\t', dtype={'ItemId':np.int64})
# 执行训练过程
iknn.fit(train_expand)

# 度量方法
mrr_20 = metric.MRR(20)
hit_20 = metric.HitRate(20)
mrr_20.init(train_expand)
hit_20.init(train_expand)

# [(metric, value), ...]
results = evaluate_sessions( iknn, [mrr_20, hit_20], test_expand, train_expand )

for e in results :
    log(log_dir, "{} : {}".format(e[0] ,  e[1]))

# 开始度量
