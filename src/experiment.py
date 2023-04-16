from dataclasses import asdict
from imports.general import *
from imports.ml import *
from src.metrics import Metrics
from src.dataset import Dataset
from src.optimizer import Optimizer
from .parameters import Parameters
from src.recalibrator import Recalibrator



class Experiment(object):
    def __init__(self, parameters: Parameters) -> None:
        self.__dict__.update(asdict(parameters))
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.dataset = Dataset(parameters)
        self.optimizer = Optimizer(parameters)
        self.metrics = Metrics(parameters)
        self.optim_method = parameters.optim_method

    def __str__(self):
        return (
            "Experiment:"
            + self.dataset.__str__
            + "\r\n"
            + self.optimizer.__str__
            + "\r\n"
            + self.metrics.__str__
        )

    def run(self) -> None:

        # Epoch 0
        self.optimizer.fit_surrogate(self.dataset)
        recalibrator = (
            Recalibrator(
                self.dataset, self.optimizer.surrogate_object, mode=self.recal_mode,
            )
            if self.recalibrate
            else None
        )
        self.metrics.analyze(
            self.optimizer.surrogate_object,
            self.dataset,
            recalibrator=recalibrator,
            extensive=True,
        )
        if self.bo:
            # Epochs > 0
            for e in tqdm(range(self.n_evals), leave=False):

                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
                if self.optim_method != "Grad":
                    # BO iteration
                    x_next, acq_val, i_choice = self.optimizer.bo_iter(
                        self.dataset,
                        X_pool=self.dataset.data.X_pool,
                        recalibrator=recalibrator,
                        return_idx=True,
                    )
                    y_next = self.dataset.data.y_pool[[i_choice]]
                    f_next = (
                        self.dataset.data.f_pool[[i_choice]]
                        if not self.dataset.data.real_world
                        else None
                    )                    
                else:
                    x_next, acq_val, i_choice = self.optimizer.bo_iter(
                        self.dataset,
                        X_pool=self.dataset.data.X_pool,
                        recalibrator=recalibrator,
                        return_idx=True,
                    )
                    x_next_unscaled = x_next[0].detach().numpy()*self.dataset.data.X_std_pool_scaling + self.dataset.data.X_mean_pool_scaling
                    f_next = self.dataset.data.problem.evaluate(x_next_unscaled)
                    y_next = f_next + np.random.normal(loc=0, scale=self.dataset.data.noise_std, size=1)
                    x_next, y_next, f_next = self.dataset.data.standardize([x_next_unscaled], [y_next], [[f_next]])

                # add to dataset
                self.dataset.add_data(x_next, y_next, f_next, i_choice=i_choice)

                # Update dataset
                self.dataset.update_solution()

                # Update surrogate
                self.optimizer.fit_surrogate(self.dataset)

                if self.analyze_all_epochs:
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics or e == self.n_evals - 1,
                    )

            if not self.analyze_all_epochs:
                self.metrics.analyze(
                    self.optimizer.surrogate_object,
                    self.dataset,
                    recalibrator=recalibrator,
                    extensive=True,
                )
        else:
            if self.analyze_all_epochs:
                for e in tqdm(range(self.n_evals), leave=False):
                    X, y, f = self.dataset.data.sample_data(n_samples=1)
                    self.dataset.add_data(X, y, f)
                    self.optimizer.fit_surrogate(self.dataset)
                    recalibrator = (
                        Recalibrator(
                            self.dataset,
                            self.optimizer.surrogate_object,
                            mode=self.recal_mode,
                        )
                        if self.recalibrate
                        else None
                    )
                    self.metrics.analyze(
                        self.optimizer.surrogate_object,
                        self.dataset,
                        recalibrator=recalibrator,
                        extensive=self.extensive_metrics or e == self.n_evals - 1,
                    )
            else:
                X, y, f = self.dataset.data.sample_data(self.n_evals)
                self.dataset.add_data(X, y, f)
                self.optimizer.fit_surrogate(self.dataset)
                recalibrator = (
                    Recalibrator(
                        self.dataset,
                        self.optimizer.surrogate_object,
                        mode=self.recal_mode,
                    )
                    if self.recalibrate
                    else None
                )
            self.metrics.analyze(
                self.optimizer.surrogate_object,
                self.dataset,
                recalibrator=recalibrator,
                extensive=True,
            )

        self.dataset.save()
        self.metrics.save()


if __name__ == "__main__":
    Experiment()
