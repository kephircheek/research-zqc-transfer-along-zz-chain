import numpy as np

from quanty.basis import ComputationBasis
from quanty.geometry import ZigZagChain
from quanty.hamiltonian import XXZ
from quanty.model.homo import Homogeneous
from quanty.problem.transfer import TransferZQCAlongChain
from quanty.task.transfer_ import FitTransferZQCPerfectlyTask, TransferZQCPerfectlyTask

from loss import loss_frobenius

width = 1.3
h_angle = np.pi / 2
length = 13
n_sender = 3
n_ancillas = 1
excitations = 2
norm = False
transmission_time = 101

loss_function = loss_frobenius
method = "brute_random"
method_kwargs = {"maxiter": 5}

geometry = ZigZagChain.from_two_chain(2, width)  # two is True
model = Homogeneous(geometry, h_angle=h_angle, norm_on=(0, 1) if norm else None)
hamiltonian = XXZ(model)


class TransferZQCAlongChain_(TransferZQCAlongChain):
    def _is_extra_element(self, i, j):
        width = len(ComputationBasis(self.sender_basis.n, excitations=1))
        return (
            (i != j)
            and (i < width)
            and (j < width)
            and (((i + 1) == width) or ((j + 1) == width))
        )

    def swap_corners(self, rho):
        rho_ = rho.copy()
        width = len(ComputationBasis(self.sender_basis.n, excitations=1))
        rho_[0, 0], rho_[width - 1, width - 1] = rho_[width - 1, width - 1], rho_[0, 0]
        return rho_


problem = TransferZQCAlongChain_.init_classic(
    hamiltonian,
    length=length,
    n_sender=n_sender,
    excitations=n_sender if excitations is None else excitations,
    n_ancillas=n_ancillas,
)


transfer_task = TransferZQCPerfectlyTask(problem, transmission_time=transmission_time)

fit_transfer_task = FitTransferZQCPerfectlyTask(
    transfer_task,
    loss_function=loss_function,
    method=method,
    method_kwargs=method_kwargs,
    polish=True,
    fsolve=True,
    history_maxlen=1,
)

fit_transfer_result = fit_transfer_task.run()
result = fit_transfer_result.history[-1]
