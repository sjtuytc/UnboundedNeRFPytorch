_base_ = './tt_default.py'

expname = 'oct22_dvgo_Playground_unbounded_baseline'

vis = dict(
    height_rate = 0.6 # camera direction frustrum height
)
data = dict(
    datadir='./data/tanks_and_temples/tat_intermediate_Playground',
)

fine_train = dict(
    N_iters=40000,  # a quick validation
)