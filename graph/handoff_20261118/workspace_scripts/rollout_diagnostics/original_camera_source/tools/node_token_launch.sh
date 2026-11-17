#!/usr/bin/env bash
# Shared path setup and recipe logging for the Graph node-token launchers.

node_token_launch_usage() {
    cat <<'USAGE'
Usage: bash LAUNCHER.sh [SEED] [options]

The three joint launchers require SEED 0, 1 or 2 as the first argument.
The two three-seed launchers do not take a seed argument.

Options:
  --experiment-root PATH  Run root (defaults to this launcher's server path)
  --source-root PATH      Checkout containing main.py (default: RUN_ROOT/source)
  --python PATH           Python executable (default: server continuation venv)
  --runtime-root PATH     Extra Python modules (default: server runtime; empty disables)
  --logs PATH             Log directory (default: RUN_ROOT/logs)
  -h, --help              Show this help without starting jobs

Existing recipe settings, checkpoint paths and sequential/parallel order remain
in the individual launchers and YAML files. Adapt those YAML paths separately.
USAGE
}

read_node_token_seed() {
    seed="$1"
    case "$seed" in
        0|1|2) ;;
        -h|--help) node_token_launch_usage; exit 0 ;;
        *) printf 'Expected seed 0, 1 or 2 as the first argument\n' >&2; exit 2 ;;
    esac
}

configure_node_token_launch() {
    experiment_root="$1"
    shift
    source_root=""
    log_root=""
    python_bin=/workspace/temporal_continuation_120/venv/bin/python
    runtime_root=/workspace/temporal_continuation_120/runtime

    while (( $# )); do
        case "$1" in
            -h|--help) node_token_launch_usage; exit 0 ;;
            --experiment-root|--source-root|--python|--runtime-root|--logs)
                if (( $# < 2 )); then
                    printf 'Missing value for %s\n' "$1" >&2
                    exit 2
                fi
                case "$1" in
                    --experiment-root) experiment_root="$2" ;;
                    --source-root) source_root="$2" ;;
                    --python) python_bin="$2" ;;
                    --runtime-root) runtime_root="$2" ;;
                    --logs) log_root="$2" ;;
                esac
                shift 2
                ;;
            *) printf 'Unknown launcher option: %s\n' "$1" >&2; exit 2 ;;
        esac
    done

    source_root="${source_root:-$experiment_root/source}"
    log_root="${log_root:-$experiment_root/logs}"
    if [[ ! -f "$source_root/main.py" ]]; then
        printf 'Source checkout must contain main.py: %s\n' "$source_root" >&2
        exit 2
    fi
    if [[ "$python_bin" == */* ]]; then
        if [[ ! -x "$python_bin" ]]; then
            printf 'Python executable is unavailable: %s\n' "$python_bin" >&2
            exit 2
        fi
        python_bin="$(cd "$(dirname "$python_bin")" && pwd -P)/$(basename "$python_bin")"
    else
        if ! python_bin="$(command -v "$python_bin")"; then
            printf 'Python executable was not found on PATH\n' >&2
            exit 2
        fi
    fi
    source_root="$(cd "$source_root" && pwd -P)"
    if [[ -n "$runtime_root" ]]; then
        if [[ ! -d "$runtime_root" ]]; then
            printf 'Runtime directory is unavailable: %s\n' "$runtime_root" >&2
            exit 2
        fi
        runtime_root="$(cd "$runtime_root" && pwd -P)"
        export PYTHONPATH="$runtime_root:$source_root"
    else
        export PYTHONPATH="$source_root"
    fi

    mkdir -p "$experiment_root" "$log_root"
    experiment_root="$(cd "$experiment_root" && pwd -P)"
    log_root="$(cd "$log_root" && pwd -P)"
    export EXPERIMENTS_DIR="$experiment_root/experiments"
    export WANDB_MODE=disabled
    cd "$source_root"
}

run_node_token_recipe() {
    # Streaming jobs retain tee + pipefail; batch jobs write only their log.
    # Callers use set -E so their ERR marker trap also runs inside this function.
    local config="$1"
    local log="$log_root/$2"
    if [[ "${3:-}" == stream ]]; then
        "$python_bin" -u main.py --config "$config" 2>&1 | tee "$log"
    else
        "$python_bin" main.py --config "$config" > "$log" 2>&1
    fi
}
