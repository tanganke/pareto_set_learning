#!/bin/bash

function run_cmd() {
    echo "--------------------------------------"
    echo $@
    echo "--------------------------------------"
    $@
}

function train() {
    run_cmd python scripts/clip_${method}.py train=true evaluate=false \
        version=${version} test_datasets=${test_datasets} model=${model} \
        lr=1e-5 num_steps=8000 save_interval=2000 \
        num_devices=${num_devices} batch_size=16
}

function evaluate() {
    run_cmd python scripts/clip_${method}.py train=false evaluate=true \
        version=${version} test_datasets=${test_datasets} model=${model} \
        num_devices=1 batch_size=64
}

function run_version() {
    # Set the terminal title
    printf "\033k${method}:${version}\033\\"

    echo "======================================="
    echo "Running method ${method} version ${version}"
    echo "======================================="
    case $version in
    0)
        model=ViT-B-32
        test_datasets="[SUN397,Cars]"
        train && evaluate
        ;;
    1)
        model=ViT-B-32
        test_datasets="[SUN397,DTD]"
        train && evaluate
        ;;
    2)
        model=ViT-B-32
        test_datasets="[Cars,DTD]"
        train && evaluate
        ;;
    3)
        model=ViT-B-32
        test_datasets="[SUN397,Cars,DTD]"
        train && evaluate
        ;;
    4)
        model=ViT-B-32
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
        train && evaluate
        ;;
    10)
        # ===== ViT-L/14, Two tasks =====
        model=ViT-L-14
        test_datasets="[SUN397,Cars]"
        train && evaluate
        ;;
    11)
        model=ViT-L-14
        test_datasets="[SUN397,DTD]"
        train && evaluate
        ;;
    12)
        model=ViT-L-14
        test_datasets="[Cars,DTD]"
        train && evaluate
        ;;
    14)
        model=ViT-L-14
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"
        run_cmd python scripts/clip_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} \
            lr=1e-5 num_steps=8000 save_interval=2000 \
            num_devices=${num_devices} batch_size=10 &&
            evaluate
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

# Parse command-line options
while (("$#")); do
    case "$1" in
    --num_devices)
        num_devices=$2
        shift 2
        ;;
    --)
        # end argument parsing
        shift
        break
        ;;
    -* | --*=)
        # unsupported flags
        echo "Error: Unsupported flag $1" >&2
        exit 1
        ;;
    *)
        # preserve positional arguments
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
