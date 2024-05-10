#!/bin/bash

function run_cmd() {
    echo "--------------------------------------"
    echo $@
    echo "--------------------------------------"
    $@
}

function train() {
    run_cmd python scripts/gpt2_pareto_${method}.py train=true evaluate=false \
        version=${version} tasks=${tasks} model=${model} partial=$partial \
        alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} init_lambda=${init_lambda} \
        num_steps=4000 save_interval=1000
}

function evaluate() {
    run_cmd python scripts/gpt2_pareto_${method}.py train=false evaluate=true batch_size=64 \
        version=${version} tasks=${tasks} model=${model} partial=$partial \
        num_steps=4000 save_interval=1000
}

function run_version() {
    # Set the terminal title
    printf "\033k${method}:${version}\033\\"

    echo "======================================="
    echo "Running method ${method} version ${version}"
    echo "======================================="
    case $version in
    3)
        batch_size=8
        partial=true
        tasks="[CoLA,MNLI]"
        init_lambda=0.6
        train && evaluate
        ;;
    4)
        batch_size=8
        partial=true
        tasks="[CoLA,MRPC]"
        init_lambda=0.6
        train && evaluate
        ;;
    5)
        batch_size=8
        partial=true
        tasks="[MRPC,MNLI]"
        init_lambda=0.6
        train && evaluate
        ;;
    6)
        batch_size=8
        partial=true
        tasks="[mrpc,mnli,cola,sst2,qnli,qqp,rte]"
        init_lambda=0.3
        train && evaluate
        ;;
    *)
        echo "Invalid version"
        exit 1
        ;;
    esac
    echo "======================================="
    echo "Finished method ${method} version ${version}"
    echo "======================================="
}

# Default values
num_devices=1
model=gpt2

# Parse command-line options
while (("$#")); do
    case "$1" in
    --num_devices)
        num_devices=$2
        shift 2
        ;;
    --) # end argument parsing
        shift
        break
        ;;
    -* | --*=) # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
    *) # preserve positional arguments
        PARAMS="$PARAMS $1"
        shift
        ;;
    esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

# Check if method and version are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <method> <version> [--num_devices n]"
    exit 1
fi

method=$1
version=$2

run_version
