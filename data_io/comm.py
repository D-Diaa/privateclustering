import sys
import time

from mpi4py import MPI

"""
This module provides a simple wrapper around the MPI communication library.
The Communicator class provides a simple interface to send and receive messages
between processes. It also provides methods to gather, allgather, and broadcast
messages to all processes. The Communicator class also provides methods to
measure communication statistics such as the number of communication rounds and the total communication size.
"""


class Communicator:
    def __init__(self, delay=0.05):
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.root = 0
        self.delay = delay

        # Create a new group excluding the root
        self.client_group = self.comm.Get_group().Excl([self.root])
        # Create a new communicator from the new group
        self.client_comm = self.comm.Create(self.client_group)

        # Initialize communication statistics
        self.num_comm_rounds = 0
        self.comm_size = 0

    def set_delay(self, delay):
        self.delay = delay

    def send(self, obj, dest, group="all"):
        if group == "all":
            self.comm.send(obj, dest=dest)
        elif group == "clients":
            self.client_comm.send(obj, dest=dest)

    def send_delay(self, obj, dest, group="all"):
        time.sleep(self.delay)
        self.num_comm_rounds += 1
        self.comm_size += obj.nbytes
        self.send(obj, dest, group)

    def recv(self, source, group="all"):
        if group == "all":
            return self.comm.recv(source=source)
        elif group == "clients":
            return self.client_comm.recv(source=source)

    def recv_delay(self, source, group="all"):
        self.num_comm_rounds += 1
        return self.recv(source, group)

    # Gather operation
    def gather(self, obj, root, group="all"):
        if group == "all":
            return self.comm.gather(obj, root=root)
        elif group == "clients":
            return self.client_comm.gather(obj, root=root)

    def gather_delay(self, obj, root, group="all"):
        self.comm_size += obj.nbytes if root != self.rank else 0
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.gather(obj, root, group)

    # Allgather operation
    def allgather(self, obj, group="all"):
        if group == "all":
            return self.comm.allgather(obj)
        elif group == "clients":
            return self.client_comm.allgather(obj)

    def allgather_delay(self, obj, group="all"):
        self.comm_size += obj.nbytes * (self.world_size - 1)
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.allgather(obj, group)

    # Broadcast operation
    def bcast(self, obj, root, group="all"):
        if group == "all":
            return self.comm.bcast(obj, root=root)
        elif group == "clients":
            return self.client_comm.bcast(obj, root=root)

    def bcast_delay(self, obj, root, group="all"):
        self.comm_size += obj.nbytes * (self.world_size - 1) if self.rank == root else 0
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.bcast(obj, root, group)

    def get_comm_stats(self):
        return {
            'num_comm_rounds': self.num_comm_rounds,
            'comm_size': self.comm_size
        }

    def print_comm_stats(self):
        to_print = f"{self.rank}, {self.get_comm_stats()}"
        print(to_print)
        sys.stdout.flush()
        return to_print

    def reset_comm_stats(self):
        self.num_comm_rounds = 0
        self.comm_size = 0

    def close(self):
        self.client_group.Free()
        if self.client_comm != MPI.COMM_NULL:
            self.client_comm.Free()


comm = Communicator()


def debug(obj=None):
    print(comm.rank, obj)
    sys.stdout.flush()


def fail_together(fn, error_message):
    feasible = True
    try:
        x = fn()
    except Exception as e:
        debug(e)
        feasible = False
    # to let all parties now if a client failed
    # Without delay cause this is not part of the protocol
    # In practice, when a party fails, everyone will timeout and ignore
    feasibles = comm.allgather(feasible)
    if not all(feasibles):
        raise Exception(error_message)
    return x
