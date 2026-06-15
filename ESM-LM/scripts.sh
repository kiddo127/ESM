
set -e pipefail

BASE_DIR="/seu_share2/home/gengxin/230238562/WorkBench/ESM-LM/discriminative_moe"

date_today=$(date '+%Y-%m-%d')
outdir=${outdir:="${BASE_DIR}/outs/merge_results"}
mkdir -p ${outdir}


models_name=(
"cola"
"sst2"
"mrpc"
"stsb"
"qqp"
"mnli"
"qnli"
"rte"
)
models_to_merge=()
for d in "${models_name[@]}"; do
models_to_merge+=(../roberta/$d/roberta-base_lr1e-05)
done
select_merge=${select_merge:="8"}


function pos(){

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1
fi
src_merge=("${models_name[@]:0:$select_merge}")

echo ">>> merged from $select_merge tasks"
echo ">>> merge ${src_merge[@]}"

data_path="${BASE_DIR}/data/test.json"
}


function esm_m(){

pos

yml='config/esm_m.yml'
select_merge=${select_merge:="8"}

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1
fi

src_merge=("${models_name[@]:0:$select_merge}")

scaling=${scaling:="1.0"}
principal_data_path=${principal_data_path:="${BASE_DIR}/data/validation.json"}
seed=${seed:="10"}
prototype_proxy_num=${prototype_proxy_num:="64"}

echo ">>> use data_path $data_path"
echo ">>> use principal_data_path $principal_data_path"
echo ">>> use seed $seed"
echo ">>> use prototype_proxy_num $prototype_proxy_num"
echo ">>> use outdir $outdir"
echo ">>> merged from $select_merge tasks"
echo ">>> use yml $yml"

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file $yml \
--exclude-param ".*classifier.*" ".*bias.*" \
--scaling $scaling \
--principal-data-path $principal_data_path \
--save-path ${save_path:="${BASE_DIR}/outs/esm_merged"} \
--prototype-proxy-num $prototype_proxy_num \
--seed $seed \
--outdir $outdir

}


function esm_r(){

    pos

    yml='config/esm_r.yml'
    select_merge=${select_merge:="8"}

    if [ $select_merge -eq 1 ]; then
        echo "please set \$select_merge > 1"
        exit 1
    fi

    src_merge=("${models_name[@]:0:$select_merge}")

    merged_model_path=${merged_model_path:=""}
    prototype_proxy_num=${prototype_proxy_num:="64"}
    seed=${seed:="10"}
    principal_data_path=${principal_data_path:="${BASE_DIR}/data/validation.json"}
    mode=${mode:-route}
    rank=${rank:-0}
    batch_size=${batch_size:-1}
    [ "$mode" = "oracle" ] && mode_flag="--mode oracle" || mode_flag=""
    [ "$mode" = "base" ] && mode_flag="--mode base" || :

    # Build optional flags
    merged_path_flag=""
    [ -n "$merged_model_path" ] && merged_path_flag="--merged-model-path $merged_model_path"

    echo ">>> use data_path $data_path"
    echo ">>> use principal_data_path $principal_data_path"
    echo ">>> use seed $seed"
    echo ">>> use prototype_proxy_num $prototype_proxy_num"
    echo ">>> use outdir $outdir"
    echo ">>> merged from $select_merge tasks"
    echo ">>> use yml $yml"
    echo ">>> mode=$mode"

    python run_merge.py \
    --models-to-merge ${models_to_merge[@]} \
    --models-name ${models_name[@]} \
    --src-merge ${src_merge[@]} \
    --data-path $data_path \
    --yaml-file $yml \
    --exclude-param ".*classifier.*" ".*bias.*" \
    --rank $rank \
    --batch-size $batch_size \
    --principal-data-path $principal_data_path \
    --prototype-proxy-num $prototype_proxy_num \
    --seed $seed \
    $merged_path_flag \
    $mode_flag \
    --outdir $outdir

}
