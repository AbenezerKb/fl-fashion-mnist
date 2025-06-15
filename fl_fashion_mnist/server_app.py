"""fl-fashion-mnist: A Flower / PyTorch app."""

from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fl_fashion_mnist.task import Net, get_weights, set_weights, test, transform_data
from datasets import load_dataset
from torch.utils.data import DataLoader
from fl_fashion_mnist.average_strategy import CustomeAverageStrategy

import json


def get_evaluate(testloader, device):

    def evaluate(server_round, parameters_ndarrays, config):

        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"centralized_accuracy": accuracy}
    
    return evaluate
        
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    values = []
    for _, m in metrics:
        metrics = m["metrics"]
        metrics_json = json.loads(metrics)
        values.append(metrics_json["m1"])
    
    return {"max m1": max(values) if values else 0.0}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * metrics["accuracy"] for num_examples, metrics in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / total_examples}

def on_fit_config(server_round: int) -> Metrics:
    """Adjust learning rate based on server round."""
    lr = 0.01
    if server_round > 2:
        lr = 0.005
    return {"learning_rate": lr}

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]

    testloader = DataLoader(testset.with_transform(transform_data()), batch_size=32, shuffle=False)

    # Define strategy
    strategy = CustomeAverageStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate(testloader, device="cpu"),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
