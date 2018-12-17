import sys
sys.path.append("./")
from _shallow_model_cknn import *
import _metric_accurate as metric
from Utils import evaluate_sessions, log

log_dir = "./_SHALLOW_MODEL.txt"
args = sys.argv
dataSetName = args[1]


log(log_dir, "-----model_name:{} data_set_name:{}".format("iknn",dataSetName))

print("data_set_name:", args[1])

cknn = ContextKNN(int(args[2]), sample_size=int(args[3]), pop_boost=1)

# load data
train_expand = pd.read_csv("./{}/data/preprocessed/train-expand.txt".format(dataSetName), sep='\t', dtype={'ItemId':np.int64})
test_expand = pd.read_csv("./{}/data/preprocessed/test-expand.txt".format(dataSetName), sep='\t', dtype={'ItemId':np.int64})
# 执行训练过程
cknn.fit(train_expand)

# 度量方法
mrr_20 = metric.MRR(20)
hit_20 = metric.HitRate(20)
mrr_20.init(train_expand)
hit_20.init(train_expand)

# [(metric, value), ...]
results = evaluate_sessions(cknn, [mrr_20, hit_20], test_expand, train_expand)

for e in results :
    log(log_dir, "{} : {}".format(e[0] ,  e[1]))

# 开始度量
