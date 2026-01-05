import random
import torch
from torch.utils.data import DataLoader, Subset
import copy
from tqdm import tqdm
import os
import json

from datasets import MNISTRegressionDataset, MNISTClassificationDataset


from builder import build_model, build_acquisiton_fn, build_optimizer
from utils import get_mnist_datasets


# runs acquire and train loop
class MNISTTrainer:
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.seed = random.randint(0, 2**31 - 1)
        self.model_cfg = cfg.model
        self.acquisition_cfg = cfg.acquisition
        self.acquisition_fn = build_acquisiton_fn(self.acquisition_cfg["function"])

        self.train_indices = list()
        self.val_indices = list()
        self.pool_indices = list()

        self.dataset_cls = MNISTClassificationDataset
        self.task = "cls"
        if hasattr(cfg, "task"):
            if cfg.task.lower() == "regression":
                self.dataset_cls = MNISTRegressionDataset
                self.task = "reg"

    def __call__(self):
        mnist_train_raw, mnist_test_raw = get_mnist_datasets()

        mnist_train = self.dataset_cls(mnist_train_raw)
        mnist_test = self.dataset_cls(mnist_test_raw)

        self.train_indices, self.val_indices, self.pool_indices = \
            self.create_initial_splits(mnist_train, train_size=20, val_size=100, num_classes=10, seed=self.seed)
        test_loader = DataLoader(   # fixed, never changes again
            mnist_test,
            batch_size=1000,
            shuffle=False,
            num_workers=0
        )

        val_loader = DataLoader(    # fixed, never changes again
            Subset(mnist_train, self.val_indices),
            batch_size=len(self.val_indices),
            shuffle=False,  # meaningless as batch is full dataset
            num_workers=0
        )

        train_loader = DataLoader(  # tb changed with acquisitions
            Subset(mnist_train, self.train_indices),
            batch_size=len(self.train_indices),
            shuffle=False,  # meaningless as batch is full dataset
            num_workers=0
        )

        test_outs = dict()

        wd_lambda = "nan"
        if hasattr(self.cfg, "weight_decay_candidates"):
            wd_lambda = self.tune_weight_decay(train_loader, val_loader)

        model = build_model(self.model_cfg)
        optimizer = build_optimizer(model.parameters(), self.cfg.optimizer)
        model, _ = self.train_once(model, train_loader, val_loader, optimizer, wd_lambda, iteration=0)

        test_outs[len(train_loader.dataset)] = self.test_once(model, test_loader, iteration=0)

        pbar = tqdm(range(1, self.acquisition_cfg["iters"] + 1),
                    desc="Active learning iterations",
                    total=self.acquisition_cfg["iters"])
        for acq_iter in pbar:
            # run acquisitions
            self.acquire(model, mnist_train)
            model = build_model(self.model_cfg)
            train_loader = DataLoader(
                Subset(mnist_train, self.train_indices),
                batch_size=len(self.train_indices),
                shuffle=False,  # meaningless as batch is full dataset
                num_workers=0
            )
            optimizer = build_optimizer(model.parameters(), self.cfg.optimizer)
            model, _ = self.train_once(model, train_loader, val_loader, optimizer, wd_lambda, iteration=acq_iter)

            train_size = len(train_loader.dataset)
            test_metrics = self.test_once(model, test_loader, iteration=acq_iter)
            test_outs[train_size] = test_metrics

            # log metrics in tqdm bar
            postfix = {
                "train_size": train_size,
                "acc": f"{test_metrics['acc']:.4f}",
            }

            if self.task == "cls":
                postfix["nll"] = f"{test_metrics['predictive_nll']:.4f}"
            elif self.task == "reg":
                postfix["rmse"] = f"{test_metrics['rmse']:.4f}"

            pbar.set_postfix(**postfix)

        # save the test_outs and model checkpoint / cfg
        base_out_dir = os.path.dirname(self.cfg.out_path)

        # append seed as subdirectory
        seed_dir = os.path.join(base_out_dir, f"seed_{self.seed}")
        os.makedirs(seed_dir, exist_ok=True)

        # save json
        out = {
            "seed": int(self.seed),
            "weight_decay_lambda": float(wd_lambda),
            "results": test_outs,  # keyed by train_size
        }
        json_path = os.path.join(seed_dir, "results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)

        # save final model
        model_path = os.path.join(seed_dir, "final_model.pt")
        torch.save(
            {
                "model_type": self.model_cfg["type"],
                "model_cfg": self.model_cfg,
                "state_dict": model.state_dict(),
                "seed": int(self.seed),
                "weight_decay_lambda": float(wd_lambda),
            },
            model_path,
        )

    def tune_weight_decay(self, train_loader, val_loader):
        wd_candidates = self.cfg.weight_decay_candidates

        best_wd = None
        best_val_loss = float("inf")

        # keep tuning deterministic given self.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        for wd in wd_candidates:
            model = build_model(self.model_cfg)
            optimizer = build_optimizer(model.parameters(), self.cfg.optimizer)

            _, val_loss = self.train_once(model, train_loader, val_loader, optimizer, wd, iteration=0)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_wd = float(wd)

        assert best_wd is not None
        return best_wd

    # trains until val loss stops improving
    def train_once(self, model, train_loader, val_loader, optimizer, wd_lambda, iteration=0):
        device = self.device
        model, best_val_loss = self.train_loop(model, train_loader, val_loader, optimizer, wd_lambda)

        if hasattr(self.cfg, "run_analytic_solution"):
            it = iter(train_loader)

            # get first (and only) batch
            x, y = next(it)
            x, y = x.to(device), y.to(device)
            model.eval()
            model.find_analytical_params(x, y)

            # assert no more data
            try:
                next(it)
                raise AssertionError("train_loader has more than one batch")
            except StopIteration:
                pass

        if hasattr(self.cfg, "vi_stage_separated"):
            it = iter(train_loader)
            all_x, all_y = next(it)
            all_x, all_y = all_x.to(device), all_y.to(device)

            model.eval()
            model.open_vi_mode(all_x, all_y)

            opt_cfg = copy.deepcopy(self.cfg.optimizer)

            """
            if self.cfg.model_type == "MatrixNormal2StageModel":
                opt_cfg["lr"] *= 10
            """

            optimizer = build_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                opt_cfg
            )
            """
            optimizer = build_optimizer(
                filter(lambda p: p.requires_grad, model.parameters()),
                self.cfg.optimizer
            )
            """
            model, best_val_loss = self.train_for_vi(model, train_loader, val_loader, optimizer, wd_lambda)

        return model, best_val_loss

    # frozen features so compute them only once
    def train_for_vi(self, model, train_loader, val_loader, optimizer, wd_lambda):
        device = self.device

        it = iter(train_loader)
        all_x, all_y = next(it)
        all_x, all_y = all_x.to(device), all_y.to(device)

        it2 = iter(val_loader)
        all_x_val, all_y_val = next(it2)
        all_x_val, all_y_val = all_x_val.to(device), all_y_val.to(device)

        # assert train and val loader has no more data
        try:
            _ = next(it)
            assert False, "train_loader has more than 1 batch; VI cache expects full-batch loader"
        except StopIteration:
            pass
        try:
            _ = next(it2)
            assert False, "val_loader has more than 1 batch; VI cache expects full-batch loader"
        except StopIteration:
            pass

        model = model.to(device)

        patience = self.cfg.patience

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        model.eval()
        with torch.no_grad():
            feats = model.forward_train(all_x, return_feats=True)
            feats_val = model.forward_train(all_x_val, return_feats=True)

        epoch = 0
        # train loop
        while True:
            model.train()
            optimizer.zero_grad()

            loss_dict = model.get_loss_dict(feats, all_y)

            total_loss = torch.tensor(0.0, device=device)
            for name, loss_info in loss_dict.items():
                if torch.is_tensor(loss_info):
                    total_loss += loss_info
                elif isinstance(loss_info, dict):
                    total_loss += loss_info["loss"]
                    info = loss_info.get("info", None)
                    if info is not None and epoch % 200 == 0:
                        print(f"Loss info at epoch {epoch}: {info}")
                else:
                    raise ValueError
            total_loss.backward()
            optimizer.step()

            # validation
            model.eval()
            with torch.no_grad():
                _, metrics = model.get_preds_and_metrics_w_feats(feats_val, all_y_val)
                val_loss = metrics["RMSE"].item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
            epoch += 1

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, best_val_loss

    def train_loop(self, model, train_loader, val_loader, optimizer, wd_lambda, iteration=0):
        device = self.device
        model = model.to(device)

        max_epochs = self.cfg.epochs
        patience = self.cfg.patience

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        # train loop
        for epoch in range(max_epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                loss_dict = model.forward_train(x, y, return_loss=True)
                total_loss = torch.tensor(0.0, device=device)
                for name, loss_info in loss_dict.items():
                    if name == "WeightDecay":
                        assert torch.is_tensor(loss_info)
                        total_loss += wd_lambda * loss_info
                    else:
                        if torch.is_tensor(loss_info):
                            total_loss += loss_info
                        elif isinstance(loss_info, dict):
                            total_loss += loss_info["loss"]
                            info = loss_info["info"]
                            if epoch % 50 == 0:
                                print(f"Loss info at epoch {epoch}: {info}")
                        else:
                            raise ValueError
                total_loss.backward()
                optimizer.step()

            # after each epoch, run validation to spot the model with best parameters
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_n = 0
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    _, metric_dict = model.forward_test(x, y, return_metrics=True)
                    total_loss = torch.tensor(0.0, device=device)
                    for name, loss in metric_dict.items():
                        if name == "WeightDecay":
                            raise ValueError
                        else:
                            total_loss += loss
                    bs = x.size(0)

                    val_loss_sum += float(total_loss.detach().cpu()) * bs
                    val_n += bs
                val_loss = val_loss_sum / val_n

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[INFO] Ended training at epoch {epoch} bc bad_epochs >= patience")
                    break

        if bad_epochs < patience:
            print(f"[WARN] Ended training bc model didn't converge inside MAX_EPOCH={max_epochs}")

        if best_state is not None:
            model.load_state_dict(best_state)
        else:
            print("Weird error: training didn't improve performance :((")

        return model, best_val_loss

    def test_once(self, model, test_loader, iteration=0):
        device = self.device
        model = model.to(device)
        model.eval()

        correct = 0
        n = 0

        rmse_sum = 0.0
        nll_sum = 0.0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                mean_preds, metric_dict = model.forward_test(x, y, return_metrics=True)

                bs = x.size(0)

                if self.task == "cls":
                    pred = mean_preds.argmax(dim=1)
                    correct += int((pred == y).sum().item())
                    nll_sum += float(metric_dict["PredictiveNLL"]) * bs

                elif self.task == "reg":
                    pred = mean_preds.argmax(dim=1)
                    gt = y.argmax(dim=1)
                    correct += int((pred == gt).sum().item())
                    rmse_sum += float(metric_dict["RMSE"]) * bs

                n += bs

        out = {
            "iter": int(iteration),
            "n": int(n),
            "acc": float(correct / n),
        }

        if self.task == "cls":
            out["predictive_nll"] = float(nll_sum / n)

        elif self.task == "reg":
            out["rmse"] = float(rmse_sum / n)

        return out

    # run model for the pool elements,
    # assign a score to each of them
    # add the indices of the ones having top scores to train indices, removing from pool indices
    def acquire(self, model, dataset):
        model.eval()

        num_points_to_acquire = self.acquisition_cfg["to_acq_per_iter"]
        bs = 1000

        pool = Subset(dataset, self.pool_indices)
        pool_loader = DataLoader(
            pool,
            batch_size=bs,
            shuffle=False,
            num_workers=0
        )

        scores = []  # acquisitions scores (aligned with pool_indices order)
        with torch.no_grad():
            for x, _ in pool_loader:
                x = x.to(next(model.parameters()).device)

                # MC dropout forward if BCNN, else normal forward
                mean_preds, all_probs = model.forward_to_acquire(x)  # (bs,10), (T,bs,10) or (bs, 10), (predictive var, aleatoric, epistemic)

                # find acquisitions score per sample
                batch_scores = self.acquisition_fn(mean_preds, all_probs)  # (bs, )

                scores.append(batch_scores.cpu())

        scores = torch.cat(scores, dim=0)  # (len(pool_indices),)
        assert scores.shape[0] == len(self.pool_indices)

        # select top-K pool positions
        topk = torch.topk(scores, k=num_points_to_acquire, largest=True)
        selected_pool_positions = topk.indices.tolist()

        # map pool positions -> MNIST indices
        selected_indices = [self.pool_indices[pos] for pos in selected_pool_positions]

        # update splits
        selected_set = set(selected_indices)

        self.train_indices.extend(selected_indices)
        self.pool_indices = [i for i in self.pool_indices if i not in selected_set]

    # returns random but balanced train_indices, val_indices; the rest in pool indices
    def create_initial_splits(self, mnist_train, train_size=20, val_size=100, num_classes=10, seed=0):
        assert train_size % num_classes == 0
        assert val_size % num_classes == 0

        g = torch.Generator().manual_seed(seed)
        targets = mnist_train.targets  # tensor of 60.000

        train_per_class = train_size // num_classes
        val_per_class = val_size // num_classes

        train_idx = []
        val_idx = []

        for c in range(num_classes):
            class_idx = torch.where(targets == c)[0]

            # shuffle deterministically with seed
            perm = class_idx[torch.randperm(len(class_idx), generator=g)]

            val_c = perm[:val_per_class]
            train_c = perm[val_per_class: val_per_class + train_per_class]

            val_idx.extend(val_c.tolist())
            train_idx.extend(train_c.tolist())

        used = {*train_idx, *val_idx}

        pool_idx = [i for i in range(len(mnist_train)) if i not in used]

        return train_idx, val_idx, pool_idx


