param(
    [string]$Python = "python",
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
& $Python ".\audit_dataset.py" --config ".\configs\full_224.json"
& $Python ".\run_experiments.py" --config ".\configs\full_224.json" --suites ablation --device $Device
& $Python ".\run_experiments.py" --config ".\configs\kan_confirmation_224.json" --suites comparison --device $Device
& $Python ".\select_kan_confirmation.py" --results-dir ".\outputs\kan_confirmation_224" --target-config ".\configs\full_224.json" --apply
& $Python ".\run_experiments.py" --config ".\configs\full_224.json" --suites comparison --device $Device
& $Python ".\visualize_experiments.py" --run-dir ".\outputs\full_224"
