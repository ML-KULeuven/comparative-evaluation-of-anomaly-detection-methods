
from .pyodflow import (KNNFlow, IFFlow, HBOSFlow, LOFFlow, CBLOFFlow, OCSVMFlow)


class BenchmarkInterface:
    anomaly_flows = dict(
        lof=LOFFlow,
        knn=KNNFlow,
        hbos=HBOSFlow,
        cblof = CBLOFFlow,
        ocsvm = OCSVMFlow,
        LOF=LOFFlow,
        KNN=KNNFlow,
        IF=IFFlow,
        IForest=IFFlow,
        HBOS=HBOSFlow,
        CBLOF = CBLOFFlow,
        OCSVM = OCSVMFlow
    )

    def __init__(
        self, anomaly_detector, experiment_identifier, flow_identifier, out_path, root_levels_up, dataset_config, algo_config, timeout_s, **kwargs
    ):

        if isinstance(anomaly_detector, str):
            self.flow = self.anomaly_flows[anomaly_detector](
                experiment_identifier, flow_identifier, out_path, root_levels_up, dataset_config, algo_config, timeout_s, **kwargs
            )
        else:
            raise NotImplementedError("this does not work anymore.")
        return

    def get_flow(self):
        return self.flow

    def unwrap(self):
        return self.get_flow()
