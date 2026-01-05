base_out_dir = "./outs"
model_type = "BCNN"
acq_method = "MaxPredictiveEntropy"

out_path = f"{base_out_dir}/{model_type.lower()}/{acq_method}/"

weight_decay_candidates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

epochs = 1000
patience = 10

model = dict(
    type=model_type,
    loss_dict={
        "CrossEntropy": dict(type="CrossEntropyLoss"),
        "WeightDecay": dict(type="L2WeightDecay"),
        # weight_decay, to be defined in train loop
    },
    T=30  # MC passes
)

acquisition=dict(
    function=dict(type=acq_method),
    to_acq_per_iter=10,
    iters=100,
)

optimizer = dict(
    type="Adam",
    lr=1e-3,
    # no weight decay, we explicitly use it
)