_base_ = './tt_default.py'

expname = 'sep26_dvgo_Playground_unbounded_debug'

data = dict(
    datadir='./data/tanks_and_temples/tat_intermediate_Playground',
)

fine_train = dict(
    N_iters=3000,  # a quick validation
)