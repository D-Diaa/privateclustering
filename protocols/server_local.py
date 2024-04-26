from tqdm import tqdm

from configs.params import Params
from parties.clients import MaskedClient, UnmaskedClient
from parties.server import Server
from utils.utils import *

"""
This file contains the server protocols for the distributed clustering algorithms.  
The server protocols are implemented as functions that take in the value lists and parameters and return the final
centroids.
"""
"""
Local Server Protocols
"""


def local_server_sumcount(value_lists, params: Params, method="masked"):
    params.dp_release = "sumcount"
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    server = Server(params)
    pbar = tqdm(range(params.iters))
    for _ in pbar:
        totals = []
        counts = []
        for client in clients:
            client.compute_assoc()
            total, count = client.step_sumcount()
            totals.append(total)
            counts.append(count)
        total, count = server.step_sumcount(totals, counts)
        for client in clients:
            client.update_centroids_sumcount(total, count)
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
    return to_fixed(centroids)


def local_server_centroid(value_lists, params: Params, method="masked"):
    params.dp_release = "centroid"
    set_seed(params.seed)
    cls = MaskedClient if method == "masked" else UnmaskedClient
    clients = [
        cls(client, value_lists[client], params)
        for client in range(params.num_clients)
    ]
    centroids = clients[0].centroids
    server = Server(params)
    pbar = tqdm(range(params.iters))
    for _ in pbar:
        scaled_cents = []
        for client in clients:
            client.compute_assoc()
            scaled_cent = client.step_centroid()
            scaled_cents.append(scaled_cent)
        new_centroids = server.step_centroid(scaled_cents)
        for client in clients:
            client.update_centroids_centroid(new_centroids)
        err = np.linalg.norm(clients[0].centroids - centroids)
        pbar.set_description(str(err))
        centroids = clients[0].centroids
    return to_fixed(centroids)
