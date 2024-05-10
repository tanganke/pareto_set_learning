#!/bin/bash

function run_cmd() {
    echo "--------------------------------------"
    echo $@
    echo "--------------------------------------"
    $@
}

function train() {
    run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
        version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
        alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
}

function evaluate() {
    run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true batch_size=64 \
        version=${version} test_datasets=${test_datasets} model=${model} partial=$partial
}

function run_version() {
    # Set the terminal title
    printf "\033k${method}:${version}\033\\"

    echo "======================================="
    echo "Running method ${method} version ${version}"
    echo "======================================="
    case $version in
    0)
        # ===== ViT-B/32, Two-task, MLP-only, init_lambda=0.3 =====
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars]"

        train
        evaluate
        ;;
    1)
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,DTD]"

        train
        evaluate
        ;;
    2)
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[Cars,DTD]"

        train
        evaluate
        ;;
    3)
        # ===== ViT-B/32, Three-task, MLP-only, init_lambda=0.3 =====
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars,DTD]"

        train
        evaluate
        ;;
    4)
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        # train
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=2 batch_size=64
        ;;
    5)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars]"

        train
        evaluate
        ;;
    6)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,DTD]"

        train
        evaluate
        ;;
    7)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[Cars,DTD]"

        train
        evaluate
        ;;
    8)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars,DTD]"

        train
        evaluate
        ;;
    9)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        train
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=equal_weight batch_size=64
        ;;
    10)
        # ===== ViT-L/14, two tasks, MLP only, init_lambda=0.3 =====
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars]"

        train
        evaluate
        ;;
    11)
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[SUN397,DTD]"

        train && evaluate
        ;;
    12)
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[Cars,DTD]"

        train && evaluate
        ;;
    14)
        model=ViT-L-14
        partial=true
        batch_size=12
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            alpha=1 lr=1e-2 num_devices=4 batch_size=${batch_size} num_steps=4000
        # evaluate
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=equal_weight batch_size=64
        ;;
    15)
        # ===== ViT-L/14, Two tasks, All-layer, init_lambda=0.3 =====
        model=ViT-L-14
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars]"

        train && evaluate
        ;;
    16)
        model=ViT-L-14
        partial=false
        batch_size=16
        test_datasets="[SUN397,DTD]"

        # train &&
        evaluate
        ;;
    17)
        model=ViT-L-14
        partial=false
        batch_size=16
        test_datasets="[Cars,DTD]"

        train && evaluate
        ;;
    19)
        # ===== ViT-L/14, Eight Tasks, All-layer, init_lambda=0.3 =====
        model=ViT-L-14
        partial=false
        batch_size=8
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            alpha=1 lr=1e-2 num_devices=4 batch_size=${batch_size} num_steps=4000
        # evaluate
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=equal_weight batch_size=64
        ;;
    20)
        # ===== ViT-B/32, Two-task, MLP-only, init_lambda=0.6 =====
        # set init_lambda to 0.6
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=1 batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    21)
        # set init_lambda to 0.6
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    22)
        # set init_lambda to 0.6
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[Cars,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    23)
        # ===== ViT-B/32, Two-task, All-layer, init_lambda=0.6 =====
        # set init_lambda to 0.6, partial=false
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    24)
        # set init_lambda to 0.6, partial=false
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    25)
        # set init_lambda to 0.6, partial=false
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[Cars,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    26)
        # ===== ViT-L/14, Two-task, MLP-only, init_lambda=0.6 =====
        # set init_lambda to 0.6, partial=true, ViT-L-14
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    27)
        # set init_lambda to 0.6, partial=true, ViT-L-14
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[SUN397,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    28)
        # set init_lambda to 0.6, partial=true, ViT-L-14
        model=ViT-L-14
        partial=true
        batch_size=16
        test_datasets="[Cars,DTD]"
        # train
        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        evaluate
        ;;
    29)
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=equal_weight batch_size=64
        ;;
    30)
        model=ViT-B-32
        partial=false
        batch_size=16
        test_datasets="[SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD]"

        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000
        run_cmd python scripts/clip_pareto_${method}.py train=false evaluate=true \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial \
            num_evaluation_samples=equal_weight batch_size=64
        ;;
    31)
        # ===== ViT-B/32, Three-task, MLP-only, init_lambda=0.3 =====
        model=ViT-B-32
        partial=true
        batch_size=16
        test_datasets="[SUN397,Cars,DTD]"

        run_cmd python scripts/clip_pareto_${method}.py train=true evaluate=false \
            version=${version} test_datasets=${test_datasets} model=${model} partial=$partial init_lambda=0.6 \
            alpha=1 lr=1e-2 num_devices=${num_devices} batch_size=${batch_size} num_steps=4000 &&
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
