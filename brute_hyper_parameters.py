import argparse
import itertools
import json
import pathlib
from dataclasses import dataclass
import shutil

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


def parse_number(n: str):
    return int(n) if float(n) == int(float(n)) else float(n)

def parse_slice(s: str):
    args = [parse_number(n) if n.strip() else None for n in s.split(":")]
    return slice(*args)

def dump_slice(a, b, n, ndigits=2):
    return f"{round(a, ndigits)}_{round(b, ndigits)}_{n}"


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
    parser.add_argument("--tt", type=parse_slice, default=None)
    parser.add_argument("--width", type=parse_slice, default=None)
    parser.add_argument("--angle", type=parse_slice, default=None)
    args = parser.parse_args()

    ta_default = 0
    tb_default = args.length * 10
    if args.tt is None:
        ta = ta_default
        tb = tb_default
        tn = int(abs(ta - tb) / 10) + 1
    else:
        ta = args.tt.start or ta_default
        tb = args.tt.stop or tb_default
        tn = args.tt.step or int(abs(ta - tb) / 10) + 1

    wa_default = 0
    wb_default = 3
    if args.width is None:
        wa = wa_default
        wb = wb_default
        wn = int(abs(wa - wb) * 10) + 1
    else:
        wa = args.width.start or wa_default
        wb = args.width.stop or wb_default
        wn = args.width.step or int(abs(wa - wb) * 10) + 1

    aa_default = 0
    ab_default = np.pi
    if args.angle is None:
        aa = aa_default
        ab = ab_default
        an = int(abs(aa - ab) * 10) + 1
    else:
        aa = args.angle.start or aa_default
        ab = args.angle.stop or ab_default
        an = args.angle.step or int(abs(aa - ab) * 10) + 1

    tt_suffix = f"{dump_slice(ta, tb, tn)}"
    width_suffix = f"{dump_slice(wa, wb, wn)}"
    angle_suffix = f"{dump_slice(aa, ab, an)}"
    norm_suffix = '-norm' if args.norm else ''
    path_name = f"n{args.length}-tt_{tt_suffix}-w_{width_suffix}-a_{angle_suffix}{norm_suffix}"

    path_dir_data = PATH_DIR_DATA / pathlib.Path(__file__).stem

    path_dir = path_dir_data / path_name
    (path_dir.exists() and path_dir.is_dir()) or path_dir.mkdir(parents=True)
    print(path_dir)

    tt_span = np.linspace(ta, tb, tn)
    width_span = np.linspace(wa, wb, wn)
    h_angle_span = np.linspace(aa, ab, an)

    spec_suffix = ""
    try:
        calc(
            path_dir,
            args.length,
            width_span,
            h_angle_span,
            tt_span,
            norm=args.norm,
            n_jobs=args.n_jobs,
            backend=args.backend,
        )
    except KeyboardInterrupt:
        spec_suffix = "-broken"

    results = [quanty.json.loads(p.read_text()) for p in path_dir.iterdir()]
    path = path_dir_data / f"{path_name}{spec_suffix}.json"
    path.write_text(quanty.json.dumps(results, indent=2))

    shutil.rmtree(path_dir)

    print(path)
