from tqdm import tqdm

from configs.params import Params
from data_io.comm import comm, fail_together
from parties.clients import MaskedClient, UnmaskedClient
from parties.server import Server
from utils.utils import *

"""
This file contains the server protocols for the distributed clustering algorithms.  
The server protocols are implemented as functions that take in the value lists and parameters and return the final
centroids.
"""
"""
MPI Server Protocols
"""


def mpi_server_sumcount(value_lists, params: Params, method="masked"):
    params.dp_release = "sumcount"
    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    server_process = (comm.rank == comm.root)

    def initialize_server():
        return Server(params)

    def initialize_client():
        cls = MaskedClient if method == "masked" else UnmaskedClient
        return cls(comm.rank - 1, value_lists[comm.rank - 1], params)

    if server_process:
        server = fail_together(initialize_server, "Server Initialization Failure")
    else:
        client = fail_together(initialize_client, "Client Initialization Failure")

    for _ in tqdm(range(params.iters)) if server_process else range(params.iters):
        if not server_process:
            fail_together(client.compute_assoc, "Association Failure")
            total, count = client.step_sumcount()

            # Concatenating total and count and gathering them
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.gather_delay(total_count, root=comm.root)

            # Receiving total and count, reshaping them to original shape, and updating centroids
            total_count = comm.bcast_delay(None, root=comm.root)

            total, count = np.split(total_count, [params.k * params.dim])
            total = total.reshape((params.k, params.dim))
            count = count.reshape(params.k)
            client.update_centroids_sumcount(total, count)

        if server_process:
            fail_together(lambda: True, "Association Failure")

            # Gathering total and count, splitting them, and updating the server
            total_counts = comm.gather_delay(None, root=comm.root)
            total_counts = [np.split(tc, [params.k * params.dim]) for tc in total_counts[1:]]
            totals, counts = zip(*[(total.reshape((params.k, params.dim)), count.reshape(params.k))
                                   for total, count in total_counts])
            total, count = server.step_sumcount(totals, counts)

            # Broadcasting total and count
            total_count = np.concatenate((total.flatten(), count.flatten()))
            comm.bcast_delay(total_count, root=comm.root)
    comm.print_comm_stats()
    centroids = comm.bcast(client.centroids if not server_process else None, root=1)
    return to_fixed(centroids)


def mpi_server_centroid(value_lists, params: Params, method="masked"):
    params.dp_release = "centroid"
    set_seed(params.seed)
    comm.reset_comm_stats()
    comm.set_delay(params.delay)
    server_process = (comm.rank == comm.root)

    def initialize_server():
        return Server(params)

    def initialize_client():
        cls = MaskedClient if method == "masked" else UnmaskedClient
        return cls(comm.rank - 1, value_lists[comm.rank - 1], params)

    if server_process:
        server = fail_together(initialize_server, "Server Initialization Failure")
    else:
        client = fail_together(initialize_client, "Client Initialization Failure")

    for _ in tqdm(range(params.iters)) if server_process else range(params.iters):
        if not server_process:
            fail_together(client.compute_assoc, "Association Failure")
            cents = client.step_centroid()

            comm.gather_delay(cents, root=comm.root)

            cents = comm.bcast_delay(None, root=comm.root)
            client.update_centroids_centroid(cents)

        if server_process:
            fail_together(lambda: True, "Association Failure")

            # Gathering total and count, splitting them, and updating the server
            cents = comm.gather_delay(None, root=comm.root)
            cents = cents[1:]
            cents = server.step_centroid(cents)
            # Broadcasting
            comm.bcast_delay(cents, root=comm.root)
    comm.print_comm_stats()
    centroids = comm.bcast(client.centroids if not server_process else None, root=1)
    return to_fixed(centroids)
