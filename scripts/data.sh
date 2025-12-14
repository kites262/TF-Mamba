#!/bin/bash
set -e

base_url="https://engineering.case.edu/sites/default/files/"
target_dir="data/"

declare -A DATASETS=(
  [0]="97.mat 98.mat 99.mat 100.mat"        # normal
  [1]="118.mat 119.mat 120.mat 121.mat"     # B007
  [2]="105.mat 106.mat 107.mat 108.mat"     # IR007
  [3]="130.mat 131.mat 132.mat 133.mat"     # OR007_6
  [4]="185.mat 186.mat 187.mat 188.mat"     # B014
  [5]="169.mat 170.mat 171.mat 172.mat"     # IR014
  [6]="197.mat 198.mat 199.mat 200.mat"     # OR014_6
  [7]="222.mat 223.mat 224.mat 225.mat"     # B021
  [8]="209.mat 210.mat 211.mat 212.mat"     # IR021
  [9]="234.mat 235.mat 236.mat 237.mat"     # OR021_6
)

download_group () {
    local group="$1"
    local files="$2"

    local out_dir="${target_dir}${group}"
    mkdir -p "$out_dir"

    for file in $files; do
        wget -nc "${base_url}${file}" -P "$out_dir"
    done
}

for group in "${!DATASETS[@]}"; do
    echo "Downloading group: $group"
    download_group "$group" "${DATASETS[$group]}"
done
