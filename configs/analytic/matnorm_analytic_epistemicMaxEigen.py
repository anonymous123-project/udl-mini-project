base_out_dir = "./outs"
model_type = "MatrixNormalAnalyticModel"
acq_method = "EpistemicMaxEigen"

task = "regression"

weight_decay_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

run_analytic_solution = True

out_path = f"{base_out_dir}/{model_type.lower()}/{acq_method}/"

epochs = 1500
patience = 20

model = dict(
    type=model_type,
    loss_dict={
        "MSE": dict(type="MSELoss"),
        "WeightDecay": dict(type="L2WeightDecay"),
    },
    # prior_column_cov_eq_lik_cov=prior_column_cov_eq_lik_cov, # forcefully true
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