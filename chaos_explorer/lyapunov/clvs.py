from .core import BennetinStepper, BennetinObserver

import numpy as np
import xarray as xr
from pathlib import Path
from loguru import logger
import sys
import shutil


def compute_clvs(
    rhs,
    jacobian,
    ic,
    parameters={},
    save_location="lyapunov_data",
    save_blv=True,
    save_ftble=True,
    tau=0.1,
    blv_tranient_len=1000,
    clv_transient_steps=1000,
    clv_observation_steps=100,
    block_size=100,
    method="RK45",
):
    # Logger/Folder Setup
    logger.remove()
    logger.add(sys.stdout, colorize=False, format="{time:YYYYMMDDHHmmss}|{level}|{message}")
    tmp_folder = Path("tmp/")
    tmp_folder.mkdir(parents=True, exist_ok=True)
    clv_folder = Path(save_location)
    clv_folder.mkdir(parents=True, exist_ok=True)

    # Get Bennetin Classes
    Q_ic = np.eye(len(ic)) * 1.0e-6
    bennetin_stepper = BennetinStepper(rhs, ic, jacobian, Q_ic, tau=tau, parameters=parameters, method=method)
    bennetin_observer = BennetinObserver(bennetin_stepper, quiet=True)

    # Step 0: Run BLV Transient
    logger.info("Starting BLV Transient.")
    bennetin_stepper.many_steps(blv_tranient_len)
    bennetin_stepper.time = 0

    # Step 1: Store Q and R Matrices for t1 -> t2 where we will later observe CLVS
    logger.info("BLV Transient Done. Running Ginelli Algorithm Forward Steps.")
    save_folder = tmp_folder / "clv_observation/"
    save_folder.mkdir(parents=True, exist_ok=True)
    bennetin_observer.make_observations_in_blocks(save_folder, clv_observation_steps, block_size, timer=False)

    # Step 2: Store R Matrices for t2 -> t3 for CLV transient
    bennetin_observer.store_Q = False
    save_folder = tmp_folder / "clv_convergence/"
    save_folder.mkdir(parents=True, exist_ok=True)
    bennetin_observer.make_observations_in_blocks(save_folder, clv_transient_steps, block_size, timer=False)

    # Step 3: CLV Convergence Steps, t3 -> t2
    logger.info("Forward Steps Done. Running Ginelli Algorithm Backward Steps.")
    R_file_list = list((tmp_folder / "clv_convergence/").glob("*.nc"))
    R_file_list.sort(key=lambda path: -int(path.name.split(".")[0]))  # Sort R files in reverse order

    A = np.eye(bennetin_stepper.ndim)  # Initialise random matrix to push with R^-1s
    for file in R_file_list:
        ds = xr.open_dataset(file)
        for step in np.flip(ds.time.values):
            R = ds.R.sel(time=step)
            pushedA = np.linalg.solve(R, A)  # Push A with R^-1
            norms = np.linalg.norm(pushedA, axis=0, ord=2)  # Prevent vector growth
            A = pushedA / norms

    # Step 4: CLV Observation Steps, t3 -> t2
    R_file_list = list((tmp_folder / "clv_observation/").glob("*.nc"))
    R_file_list.sort(key=lambda path: -int(path.name.split(".")[0]))  # Sort R files in reverse order

    for file in R_file_list:
        ds = xr.open_dataset(file)
        clv_ts = []
        ftcle_ts = []
        blv_ts = []
        ftble_ts = []
        for step in np.flip(ds.time.values):
            # Get Q and R
            Q = ds.Q.sel(time=step)
            R = ds.R.sel(time=step)

            # Compute and store FTCLE/CLV
            clv = np.matmul(Q.values, A)
            reversed_ftcle = -np.log(norms) / (bennetin_stepper.tau)
            clv_ts.append(clv)
            ftcle_ts.append(np.flip(reversed_ftcle))

            # Compute and store FTBLE/BLV
            blv = Q.values
            reversed_ftble = np.log(np.diag(R.values)) / (tau)
            blv_ts.append(blv)
            ftble_ts.append(np.flip(reversed_ftble))

            # Push A with R^-1
            pushedA = np.linalg.solve(R, A)
            norms = np.linalg.norm(pushedA, axis=0, ord=2)  # Prevent vector growth
            A = pushedA / norms

        # Save CLV/FTCLE files
        dic = {}
        dic['trajectory'] = ds['trajectory']
        dic["CLV"] = xr.DataArray(
            np.flip(clv_ts),
            dims=["time", "le_index", "component"],
            name="CLV",
            coords={"time": ds.time, "le_index": ds.le_index, "component": ds.component},
        )
        dic["FTCLE"] = xr.DataArray(
            np.flip(ftcle_ts),
            dims=["time", "le_index"],
            name="FTCLE",
            coords={"time": ds.time, "le_index": ds.le_index},
        )
        if save_blv:
            dic["BLV"] = xr.DataArray(
                np.flip(blv_ts),
                dims=["time", "le_index", "component"],
                name="BLV",
                coords={"time": ds.time, "le_index": ds.le_index, "component": ds.component},
            )
        if save_ftble:
            dic["FTBLE"] = xr.DataArray(
                np.flip(ftble_ts),
                dims=["time", "le_index"],
                name="FTBLE",
                coords={"time": ds.time, "le_index": ds.le_index},
            )
        ds = xr.Dataset(dic)
        ds.to_netcdf(clv_folder / file.name)
    logger.info(f"Results saved at {clv_folder}.")

    # Clean Up
    shutil.rmtree(tmp_folder)
