import argparse
import itertools
import json
import pathlib
import time

import numpy as np
import timeti

from quanty import matrix
from quanty.basis import ComputationBasis
from quanty.geometry import ZigZagChain
from quanty.hamiltonian import XXZ
from quanty.model.homo import Homogeneous
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer import ZeroCoherenceTransfer

from config import PATH_DIR_DATA

def loss_frobenius_qip2022(m):
    """Return loss function based on Frobenius norm (not true)"""
    diff = m.copy()
    diff[-1, -1] -= 1
    return -np.real(np.sqrt((diff.conj() @ diff).trace()))


def frobenius_norm(m):
    return np.sum(np.abs(m) ** 2)

def loss_frobenius(m):
    """Return loss function based on Frobenius norm (true)"""
    m_inf = np.zeros_like(m)
    m_inf[-1, -1] = 1
    return - frobenius_norm(m - m_inf)


def main(lengthes, n_sender, n_ancillas_list, loss_function):
    results_path = PATH_DIR_DATA / pathlib.Path(f"results_{int(time.time())}.json")
    results_path.touch()
    results_path.write_text("[]")
    print("Path to result:", results_path)

    geometry = ZigZagChain(angle=np.pi / 6, ra=np.sqrt(3) / 3)
    model = Homogeneous(geometry)
    hamiltonian = XXZ(model)

    method, method_kwargs = "brute_random", {"no_local_search": False, "maxiter": 2}

    for lenght, n_ancillas in timeti.profiler(
        itertools.product(lengthes, n_ancillas_list)
    ):
        print("", "-" * 80, "", sep="\n")
        task = ZeroCoherenceTransfer.init_classic(
            hamiltonian,
            length=lenght,
            n_sender=n_sender,
            n_ancillas=n_ancillas,
            excitations=2,
        )

        task.info()
        print("[fit transmission time]")
        with timeti.profiler("fit transmission time"):
            task.fit_transmission_time(log10_dt=-1)

        print("[fit transfer]")
        with timeti.profiler("fit transfer"):
            task.fit_transfer(
                loss=loss_function,
                method=method,
                method_kwargs=method_kwargs,
                fsolve=False,
                polish=False,
            )

        residual_max = np.max(np.abs(task.perfect_transferred_state_residuals()))
        sender_params = task.perfect_transferred_state_params()

        results = json.loads(results_path.read_text())
        results.append(
            {
                "task": task.dict,
                "fit_transfer_kwargs": {
                    "loss": loss_function.__name__,
                    "method": method,
                    "method_kwargs": method_kwargs,
                },
                "pts": list(sender_params.values()),
                "residual_max": residual_max,
            }
        )
        results_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length", type=str, help="start:stop:step")
    parser.add_argument("-s", "--sender", type=int, help="sender size", default=3)
    parser.add_argument("-a", "--ancillas", type=int, nargs="+", default=0)
    parser.add_argument("--frobenius-true", action="store_true")

    args = parser.parse_args()

    loss_function = loss_frobenius_qip2022
    if args.frobenius_true:
        loss_function = loss_frobenius

    lengthes = list(range(*map(int, args.length.split(":"))))
    main(lengthes, args.sender, args.ancillas, loss_function)
