base_out_dir = "./outs"
model_type = "MatrixNormal2StageModel"
prior_column_cov_eq_lik_cov = False
acq_method = "PredictiveVarLogDet"

task = "regression"

weight_decay_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

vi_stage_separated = True

out_path = f"{base_out_dir}/{model_type.lower()}/prior_eq_lik_cov={prior_column_cov_eq_lik_cov}/{acq_method}/"

epochs = 3000
patience = 50

model = dict(
    type=model_type,
    loss_dict_stage_1={
        "MSE": dict(type="MSELoss"),
        "WeightDecay": dict(type="L2WeightDecay")
    },
    loss_dict_stage_2={"MatrixNormalVI": dict(type="MatrixNormalVILoss")},
    prior_column_cov_eq_lik_cov=prior_column_cov_eq_lik_cov,
)

acquisition=dict(
    function=dict(type=acq_method),
    to_acq_per_iter=10,
    iters=100,
)

optimizer = dict(
    type="Adam",
    lr=1e-3,
)