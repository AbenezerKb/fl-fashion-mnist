from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import torch
import json
from fl_fashion_mnist.task import Net, set_weights
import wandb
from datetime import datetime


class CustomeAverageStrategy(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")        
        wandb.init(
            project="fl_fashion_mnist",
            name = f"custom_strategy_{name}",
        )

    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]],
                      failures: list[tuple[ClientProxy, FitRes] | Exception],
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | int | float | str]]:
       parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
       
       
       ndarrays = parameters_to_ndarrays(parameters_aggregated)
       
       model = Net()

       set_weights(model, ndarrays)

       torch.save(model.state_dict(), f"model_round_{server_round}.pth")

       return parameters_aggregated, metrics_aggregated
    
    def evaluate(
                self, server_round: int, parameters: Parameters,
                ) -> tuple[float, dict[str, bool | bytes | int | float | str]] | None:
        
        loss, metrics = super().evaluate(server_round, parameters)

        results = {"loss":loss, **metrics}

        self.results_to_save[server_round] = results
        
        with open("results.json","w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)
        

        wandb.log({"loss": loss, **metrics, "round": server_round})
        return loss, metrics