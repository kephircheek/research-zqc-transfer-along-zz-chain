import argparse
import itertools
import json
import pathlib
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import numpy.typing as npt
import quanty.json
import timeti
import tqdm
from joblib import Parallel, delayed
from quanty import matrix
from quanty.basis import ComputationBasis
from quanty.geometry import ZigZagChain
from quanty.hamiltonian import XXZ
from quanty.matrix.norm import frobenius_norm
from quanty.model.homo import Homogeneous
from quanty.problem.transfer import TransferAlongChain, TransferZQCAlongChain
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer_ import FitTransferZQCPerfectlyTask, TransferZQCPerfectlyTask

import loss
from config import PATH_DIR_DATA


def parse_linspace(s: str):
    a, b, n = map(str.strip, s.split(","))
    return np.linspace(float(a), float(b), int(n))


def round_smart(x: float, decimals=3):
    return int(x) if np.abs(x - int(x)) < 1e-13 else np.round(x, decimals)


def dumps_linspace(arr: str):
    a = round_smart(arr[0])
    b = round_smart(arr[-1])
    steps = arr.shape[0]
    return f"{a},{b},{steps}"


def import_loss(loss_function_name):
    return getattr(loss, loss_function_name)


@dataclass(frozen=True)
class BruteHyperParameterSubTask:
    number: int
    pathdir: pathlib.Path
    task: TransferZQCPerfectlyTask | FitTransferZQCPerfectlyTask

    def run(self):
        if isinstance(self.task, TransferZQCPerfectlyTask):
            result_main = self.task.run()

        elif isinstance(self.task, FitTransferZQCPerfectlyTask):
            r = self.task.run()
            result_main = r.history[-1]

        result = BruteHyperParameterSubTaskResult(self)
        result_main_json = quanty.json.dumps(result_main, indent=2)
        result.path.write_text(result_main_json)
        return result


@dataclass(frozen=True)
class BruteHyperParameterSubTaskResult:
    task: BruteHyperParameterSubTask

    @property
    def path(self):
        return self.task.pathdir / f"{self.task.number}.json"


@dataclass(frozen=True)
class BruteHyperParameterTask:
    length: int
    width_span: npt.ArrayLike
    h_angle_span: npt.ArrayLike
    tt_span: npt.ArrayLike
    norm: bool
    n_sender: int = 3
    n_ancillas: int = 0
    excitations: int | None = None
    loss_function: Callable | None = None
    method: Literal["dual_annealing", "brute_random"] | None = None
    method_kwargs: dict | None = None
    n_jobs: int = 1
    backend: Literal["loky", "multiprocessing", "threading"] = "loky"

    def run_subtask(self, pathdir, i, width, h_angle, tt):
        geometry = ZigZagChain.from_two_chain(2, width)  # two is True
        model = Homogeneous(
            geometry, h_angle=h_angle, norm_on=(0, 1) if self.norm else None
        )
        hamiltonian = XXZ(model)
        problem = TransferZQCAlongChain.init_classic(
            hamiltonian,
            length=self.length,
            n_sender=self.n_sender,
            excitations=self.n_sender if self.excitations is None else self.excitations,
            n_ancillas=self.n_ancillas,
        )
        transfer_task = TransferZQCPerfectlyTask(problem, transmission_time=tt)

        task_main: TransferZQCPerfectlyTask | FitTransferZQCPerfectlyTask = None
        if self.loss_function is None:
            task_main = transfer_task
        else:
            task_main = FitTransferZQCPerfectlyTask(
                transfer_task,
                loss_function=self.loss_function,
                method=self.method,
                method_kwargs=self.method_kwargs,
                polish=False,
                fsolve=True,
                history_maxlen=1,
            )

        subtask = BruteHyperParameterSubTask(i, pathdir, task_main)
        subtask_result = subtask.run()
        return subtask_result

    @timeti.profiler()
    def _run(self, pathdir):
        cases = itertools.product(self.width_span, self.h_angle_span, self.tt_span)
        return Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(self.run_subtask)(pathdir, i, width, h_angle, tt)
            for i, (width, h_angle, tt) in tqdm.tqdm(list(enumerate(cases)), ncols=90)
        )

    def run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = pathlib.Path(tmpdir)
            interupted: bool = False
            broken: bool = False
            try:
                subtask_results = self._run(tmpdir)
            except KeyboardInterrupt:
                interupted = True
            except Exception as e:
                warnings.warn(str(e))
                broken = True

            result = BruteHyperParameterResult(self, interupted=interupted, broken=broken)
            results = [quanty.json.loads(p.read_text()) for p in tmpdir.iterdir()]
            result.path.write_text(quanty.json.dumps(results, indent=2))
            return result


@dataclass(frozen=True)
class BruteHyperParameterResult:
    task: BruteHyperParameterTask
    interupted: bool = False
    broken: bool = False

    @property
    def path(self):
        path_dir_results = PATH_DIR_DATA / pathlib.Path(__file__).stem
        path_dir_results.exists() or path_dir_results.mkdir()
        return path_dir_results / self.filename

    @property
    def filename(self):
        return pathlib.Path(
            f"n{self.task.length}"
            + f"-tt_{dumps_linspace(self.task.tt_span)}"
            + f"-w_{dumps_linspace(self.task.width_span)}"
            + f"-a_{dumps_linspace(self.task.h_angle_span)}"
            + f"-s{self.task.n_sender}"
            + (f"-ex{self.task.excitations}" if self.task.excitations else "")
            + (f"-na_{self.task.n_ancillas}" if self.task.n_ancillas > 0 else "")
            + (f"-m_{self.task.method}" if self.task.method else "")
            + (
                f"-lf_{self.task.loss_function.__name__}"
                if self.task.loss_function
                else ""
            )
            + ("-norm" if self.task.norm else "")
            + ("-broken" if self.broken else "")
            + ("-interupted" if self.interupted else "")
            + ".json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Find perfect transferred state."
            " Note, if excitation is not setted top excitation number"
            " equal to sender spins number."
        )
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        choices=("loky", "multiprocessing", "threading"),
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--tt", type=parse_linspace, default=None)
    parser.add_argument("--width", type=parse_linspace, default=None)
    parser.add_argument("--angle", type=parse_linspace, default=None)
    parser.add_argument("--sender", type=int, default=3)
    parser.add_argument("--n_ancillas", type=int, default=0)
    parser.add_argument("--excitations", type=int, default=None)
    parser.add_argument("--loss_function", type=import_loss, default=None)
    parser.add_argument(
        "--method", choices=["brute_random", "dual_annealing"], default=None
    )
    parser.add_argument("--method_kwargs", type=json.loads, default=None)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()

    task = BruteHyperParameterTask(
        args.length,
        args.width,
        args.angle,
        args.tt,
        n_sender=args.sender,
        excitations=args.excitations,
        loss_function=args.loss_function,
        n_ancillas=args.n_ancillas,
        method=args.method,
        method_kwargs=args.method_kwargs,
        norm=args.norm,
        n_jobs=args.n_jobs,
        backend=args.backend,
    )
    result = task.run()
    print(result.path)
