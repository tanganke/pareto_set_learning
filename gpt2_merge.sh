#!/bin/bash

function run_cmd() {
    echo "--------------------------------------"
    echo $@
    echo "--------------------------------------"
    $@
}

function run_version() {
    # Set the terminal title
    printf "\033k${method}:${version}\033\\"

    echo "======================================="
    echo "Running method ${method} version ${version}"
    echo "======================================="

    case $version in
        0)
            model=gpt2
            tasks=("mrpc" "mnli")
            run_cmd python scripts/gpt2_${method}.py --version ${version} \
                --model ${model} --tasks ${tasks[@]}
            ;;
        1)
            model=gpt2
            tasks=('mrpc' 'cola')
            run_cmd python scripts/gpt2_${method}.py --version ${version} \
                --model ${model} --tasks ${tasks[@]}
            ;;
        2)
            model=gpt2
            tasks=('mnli' 'cola')
            run_cmd python scripts/gpt2_${method}.py --version ${version} \
                --model ${model} --tasks ${tasks[@]}
            ;;
        3)
            model=gpt2
            tasks=("mrpc" "mnli" "cola" "sst2" "qnli" "qqp" "rte")
            run_cmd python scripts/gpt2_${method}.py --version ${version} \
                --model ${model} --tasks ${tasks[@]}
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

# Check if method and version are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <method> <version>"
    exit 1
fi

method=$1
version=$2

run_version
