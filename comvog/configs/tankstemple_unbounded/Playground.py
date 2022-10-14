_base_ = './tt_default.py'

expname = 'sep30_dvgo_Playground_unbounded_baseline'

vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
data = dict(
    datadir='./data/tanks_and_temples/tat_intermediate_Playground',
)

fine_train = dict(
    N_iters=30000,  # a quick validation
)