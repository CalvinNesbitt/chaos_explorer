# TODO: Wrapper for the computation of BLVs

# Set algorithm parameters
blv_tranient_len = 1000
clv_transient_steps = 100
clv_observation_steps = 102
block_size = 10
bennetin_stepper = BennetinStepper(l63, np.array([1, 1, 1]), l63_jacobian, np.eye(3) * 1.0e-6, tau=0.1)
bennetin_observer = BennetinObserver(bennetin_stepper)

# Setup
tmp_folder = Path("tmp/")
tmp_folder.mkdir(parents=True, exist_ok=True)

# Step 0: Run BLV Transient
bennetin_stepper.many_steps(blv_tranient_len)
bennetin_stepper.time = 0

# Step 1: Store Q and R Matrices for t1 -> t2 where we will later observe CLVS
save_folder = tmp_folder / "clv_observation/"
save_folder.mkdir(parents=True, exist_ok=True)
bennetin_observer.make_observations_in_blocks(save_folder, clv_observation_steps, block_size, timer=False)

# Step 2: Store R Matrices for t2 -> t3 for CLV transient
bennetin_observer.store_Q = False
save_folder = tmp_folder / "clv_convergence/"
save_folder.mkdir(parents=True, exist_ok=True)
bennetin_observer.make_observations_in_blocks(save_folder, clv_observation_steps, block_size, timer=False)

# Step 3: CLV Convergence Steps, t3 -> t2
R_file_list = list((tmp_folder / "clv_convergence/").glob("*.nc"))
R_file_list.sort(key=lambda path: -int(path.name.split(".")[0]))  # Sort R files in reverse order

A = np.eye(bennetin_stepper.ndim)
for file in R_file_list:
    ds = xr.open_dataset(file)
    R_ts = np.flip(ds.R)
    A_ts = []
    for R in R_ts:
        print(R_ts.time)
        A_ts.append(A)
        newA = np.linalg.solve(R, A)  # Push A with R^-1
        norms = np.linalg.norm(newA, axis=0, ord=2)  # Prevent vector growth
    A = A_ts[-1]

# Step 4: CLV Observation Steps, t3 -> t2
R_file_list = list((tmp_folder / "clv_observation/").glob("*.nc"))
R_file_list.sort(key=lambda path: -int(path.name.split(".")[0]))  # Sort R files in reverse order

for file in R_file_list:
    ds = xr.open_dataset(file)
    R_ts = np.flip(ds.R)
    A_ts = many_ginelli_steps(R_ts, A)
    A = A_ts[-1]
    ds["A"] = xr.DataArray(
        A_ts,
        dims=["time", "le_index", "component"],
        name="A",
        coords={"time": np.flip(ds.time), "le_index": ds.le_index, "component": ds.component},
    )
    ds.to_netcdf(file)

# Finally Get the CLVs and Clean Up
