#!/usr/bin/env bash
# Generate synthetic patients and run fidelity + config-vs-real evaluation
# for every trial directory under syn_datasets/ that contains *_config.json.
#
# Each arm's patient count comes from its TrialConfig's n_enrolled.
PYTHON="/shared/eng/chufan2/miniconda3/envs/longehr2/bin/python"
ROOT="/shared/eng/chufan2/sunlab-kmgen/kmgen_ipd"
SYN_ROOT="${ROOT}/syn_datasets"

cd "${ROOT}"

for trial_dir in "${SYN_ROOT}"/NCT*/; do
    trial_dir="${trial_dir%/}"
    trial="$(basename "${trial_dir}")"

    if ! compgen -G "${trial_dir}/*_config.json" > /dev/null; then
        echo "[${trial}] no *_config.json in ${trial_dir} — skipping"
        continue
    fi

    mapfile -t configs < <(ls "${trial_dir}"/*_config.json | sort)

    echo "=============================================================="
    echo "[${trial}] generating synthetic patients"
    echo "=============================================================="

    "${PYTHON}" synthetic_patient_generator.py \
        "${configs[@]}" \
        -o "${trial_dir}/synthetic_patients.csv" \
        --seed 42

    echo
    echo "[${trial}] evaluating"
    "${PYTHON}" evaluate.py \
        --trial-id "${trial}" \
        --synth-csv "${trial_dir}/synthetic_patients.csv"
    echo
done

echo "All trials done."
