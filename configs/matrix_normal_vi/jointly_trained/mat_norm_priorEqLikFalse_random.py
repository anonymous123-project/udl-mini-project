base_out_dir = "./outs"
model_type = "MatrixNormalModel"
prior_column_cov_eq_lik_cov = False
acq_method = "Random"

task = "regression"

out_path = f"{base_out_dir}/{model_type.lower()}/prior_eq_lik_cov={prior_column_cov_eq_lik_cov}/{acq_method}/"

epochs = 1500
patience = 20

model = dict(
    type=model_type,
    loss_dict={"MatrixNormalVI": dict(type="MatrixNormalVILoss")},
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