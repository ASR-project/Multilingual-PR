from utils.per import PhonemeErrorRate

class MetricsModule:
    def __init__(self, set_name, device) -> None:
        """
        set_name: val/train/test
        """
        self.device = device
        dict_metrics = {}
        dict_metrics['per'] = PhonemeErrorRate(compute_on_step=False).to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):

        for _, m in self.dict_metrics.items():
            # metric on current batch
            m(x, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, pl_module):

        for k, m in self.dict_metrics.items():

            # metric on all batches using custom accumulation
            metric = m.compute()
            pl_module.log(name + k, metric)

            # Reseting internal state such that metric ready for new data
            m.reset()
            m.to(self.device)