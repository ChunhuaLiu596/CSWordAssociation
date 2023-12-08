#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

source $1 
kg_name_out=$2
kg_model_out=$3
dynamic_kg=$4
mode=$5
seeds_out=($6)
i_out=$7
# seeds_out=($6 $7 $8)
RANDOM=$$

if [[ ${kg_name_out} =~ ^("swow"|"swow1rel")$ ]]; then
	echo $kg_name_out
	kg_name=$kg_name_swow
	ent_emb_paths=$ent_emb_paths_swow
	rel_emb_path=$rel_emb_path_swow
	path_embedding_path=$path_embedding_path_swow
	seeds=$seeds_swow
elif [[ ${kg_name_out} == "cpnet_swow" ]]; then
	kg_name=$kg_name_cpsw
	ent_emb_paths=$ent_emb_paths_cpsw
	rel_emb_path=$rel_emb_path_cpsw
	path_embedding_path=$path_embedding_path_cpsw
	seeds=(0 11010 19286 21527 24524)
	echo $kg_name, $path_embedding_path
fi


if [[ ${mode} =~ ^("pred"|"decode")$ ]]; then
	seeds=$seeds_out
	n_runs=${#seeds_out[@]}
fi

for ((i=0; i<${n_runs}; i++));do

	if [[ -z ${seeds} ]]; then
		[[ $i -eq 0 ]] && seed=0 || seed=$RANDOM
	else
		if [[ -z ${seeds[$i]} ]]; then
			seed=$RANDOM
		else
			seed=${seeds[$i]}
		fi
	fi

	if [[ -z ${kg_model_out} ]]; then
		kg_model=$kg_model
	else
		kg_model=$kg_model_out
	fi


	if [[ ${dynamic_kg} == "swow" ]]; then
		echo $dynamic_kg, $kg_name_out
		path_embedding_path=$path_embedding_path_swow
	elif [[ ${dynamic_kg} == "cpnet" ]]; then
		echo $dynamic_kg, $kg_name_out
		path_embedding_path=$path_embedding_path
	elif [[ ${dynamic_kg} == "cpnet_swow" ]]; then
		echo $dynamic_kg, $kg_name_out
		path_embedding_path=$path_embedding_path_cpsw
	fi

	if [[ ${mode} == "pred" ]]; then
		save_dir="./saved_models/${dataset}/${encoder}_elr${encoder_lr}_dlr${decoder_lr}_d${dropoutm}_b${batch_size}_s${seed}_g${gpu_device}_${kg_model}_a${ablation}_${kg_name}_ent${ent_emb}_p${subsample}_${i_out}"
	fi

	if [[ ${mode} != "pred" ]]; then
		echo "making dir"
		save_dir="./saved_models/${dataset}/${encoder}_elr${encoder_lr}_dlr${decoder_lr}_d${dropoutm}_b${batch_size}_s${seed}_g${gpu_device}_${kg_model}_a${ablation}_${kg_name}_ent${ent_emb}_p${subsample}_$i"
		mkdir -p ${save_dir}
	fi 

	echo ${save_dir}/train.log 	
# 
	python -u main.py \
		--mode $mode\
		--dataset $dataset \
		--inhouse $inhouse \
		--save_dir $save_dir \
		--encoder $encoder \
		--max_seq_len $max_seq_len \
		--encoder_lr $encoder_lr \
		--decoder_lr $decoder_lr \
		--batch_size $batch_size \
		--dropoutm $dropoutm \
		--gpu_device $gpu_device \
		--nprocs 20 \
		--save_model $save_model \
		--seed $seed \
		--use_cache $use_cache\
		--ent_emb $ent_emb\
		--ent_emb_paths $ent_emb_paths\
		--rel_emb_path $rel_emb_path\
		--kg_name $kg_name\
		--kg_model $kg_model\
		--subsample $subsample\
		--debug $debug\
		--mini_batch_size $mini_batch_size\
		--path_embedding_path $path_embedding_path\
		--ablation $ablation\
		--lm_sent_pool $lm_sent_pool\
		--decoder_hidden_dim $decoder_hidden_dim\
		--encoder_dim $encoder_dim\
		--encoder_dropoute $encoder_dropoute\
		--encoder_dropouti $encoder_dropouti\
		--encoder_dropouth $encoder_dropouth\
		--encoder_layer_num $encoder_layer_num\
		--max_epochs_before_stop $max_epochs_before_stop\
		--encoder_type $encoder_type\ 
		# > ${save_dir}/train.log
	 
done