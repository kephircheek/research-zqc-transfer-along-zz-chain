import argparse
import itertools
import json
import pathlib
from dataclasses import dataclass

import numpy as np
import quanty.json
import timeti
import tqdm
from joblib import Parallel, delayed

from quanty import matrix
from quanty.basis import ComputationBasis
from quanty.geometry import ZigZagChain
from quanty.hamiltonian import XXZ
from quanty.model.homo import Homogeneous
from quanty.state.coherence import coherence_matrix
from quanty.task.transfer_ import TransferAlongChain, TransferZQCPerfectlyTask

from config import PATH_DIR_DATA

def calc_case(path, length, width, h_angle, tt, norm):
    geometry = ZigZagChain.from_two_chain(2, width)  # two is True
    model = Homogeneous(geometry, h_angle=h_angle, norm_on=(0, 1) if norm else None)
    hamiltonian = XXZ(model)
    problem = TransferAlongChain.init_classic(
        hamiltonian,
        length=length,
        n_sender=3,
        excitations=3,
    )
    task = TransferZQCPerfectlyTask(problem, transmission_time=tt)
    result = task.run()
    quanty.json.dumps(result, indent=2)
    path.write_text(quanty.json.dumps(result, indent=2))


def calc(
    path_dir, length, width_span, h_angle_span, tt_span, norm, n_jobs=1, backend="loky"
):
    results = []
    cases = itertools.product(width_span, h_angle_span, tt_span)
    Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(calc_case)(path_dir / f"{i}.json", length, width, h_angle, tt, norm)
        for i, (width, h_angle, tt) in tqdm.tqdm(list(enumerate(cases)), ncols=90)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        choices=("loky", "multiprocessing", "threading"),
    )
    parser.add_argument("--length", type=int)
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--tn", type=int, default=11)
    parser.add_argument("--tb", type=int, default=None)
    parser.add_argument("--wn", type=int, default=31)
    parser.add_argument("--ab", type=float, default=np.pi / 2)
    parser.add_argument("--an", type=int, default=31)
    args = parser.parse_args()

    tb = args.tb or args.length * 10
    wn = args.wn
    path_name = f"results_n{args.length}_tb{tb}_tn{args.tn}_wn{wn}_ab{args.ab}_an{args.an}{'_norm' if args.norm else ''}"
    path_dir = PATH_DIR_DATA / pathlib.Path(path_name)
    (path_dir.exists() and path_dir.is_dir()) or path_dir.mkdir()

    width_span = np.linspace(0, 3, wn)
    h_angle_span = np.linspace(0, args.ab, args.an)
    tt_span = np.linspace(0, tb, args.tn)
    results = calc(
        path_dir,
        args.length,
        width_span,
        h_angle_span,
        tt_span,
        norm=args.norm,
        n_jobs=args.n_jobs,
        backend=args.backend,
    )

    results = [quanty.json.loads(p.read_text()) for p in path_dir.iterdir()]
    path = PATH_DIR_DATA / pathlib.Path(f"{path_name}.json")
    path.write_text(quanty.json.dumps(results, indent=2))

    import shutil

    shutil.rmtree(path_dir)

    print(path)
