# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
# user-friendly wrapper around elq_slurm_scripts/train_elq.sh

objective=$1
data=$2  # filepath to tokenized data directory (should have a `train.jsonl`)
mention_agg_type=$3 #all_avg_linear
context_length=$4
batch_size=$5
eval_batch_size=$6
use_attention=${7}
gpu_ids=$8
model_type=$9
shot=${10}
epoch=${11}  # model checkpoint to pick up from
model_data=${12}
embedding_choice=${13}

if [ "${embedding_choice}" = "" ] || [ "${embedding_choice}" = "_" ]
then
  embedding_choice='wiki'
fi


if [ "${epoch}" = "" ] || [ "${epoch}" = "_" ]
then
    epoch=-1
fi
echo $use_attention
echo "GPU is $gpu_ids"

load_saved_cand_encs=true
adversarial=false
model_size=base
mention_scoring_method=qa_linear


echo " use attetion is "${use_attention}
export PYTHONPATH=.

data_type=${data##*/}


if [ ${model_data} != '' ]
then
  model_dir="./experiments/${data_type}/${model_data}_${model_type}_${use_attention}_${mention_agg_type}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${model_size}_${mention_scoring_method}"
else
  model_dir="./experiments/${data_type}/${model_type}_${use_attention}_${mention_agg_type}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${model_size}_${mention_scoring_method}"
fi
mkdir -p model_dir

cand_enc_args=""

if [ "${data}" != "WIKI_NER"  ]
then
 if [ "${model_type}" = 'bert-base-uncased' ]
  then
    if [ "${shot}" = '-1' ]
    then
      data_path="./dataset/${data}/tokenized_uncased"
    else
      data_path="./dataset/${data}/tokenized_uncased_"${shot}
    fi
    cand_enc_args="--lowercase"
 fi
else
  echo "Data not found: ${data}"
  exit
fi

echo $data_path

if [ "${mention_agg_type}" = "none" ]
then
  all_mention_args=""
elif [ "${mention_agg_type}" = "none_no_mentions" ]
then
  all_mention_args="--no_mention_bounds"
else
  all_mention_args="--no_mention_bounds \
    --mention_aggregation_type ${mention_agg_type}"
fi


if [ "${load_saved_cand_encs}" = "true" ]
then
  echo "loading + freezing saved candidate encodings"
  cand_enc_args="--freeze_cand_enc --load_cand_enc_only ${cand_enc_args}"
fi


if [ "${context_length}" = "" ]
then
  context_length="128"
fi

model_ckpt=$model_type


if [ ${objective} = 'train' ]
then
  learning_rate=1e-5
else
  learning_rate=6e-6
fi


if [ "${epoch}" = "" ]
then
  epoch=-1
fi

dont_distribute_train_samples="--dont_distribute_train_samples"
echo $3

if [ "${objective}" = "train" ] || [ "${objective}" = "finetune" ]
then
  echo "Running ${mention_agg_type} biencoder training on ${data} dataset."


  model_path_arg=""

  if [ ${epoch} != "-1" ]
  then
     model_path_arg="--path_to_model /media/6T/yaqing/experiments/${model_data##*/}/${model_type}_${use_attention}_${mention_agg_type}_${context_length}_${load_saved_cand_encs}_${adversarial}_bert_${model_size}_${mention_scoring_method}/epoch_${epoch}/pytorch_model.bin"
     #model_path_arg='--path_to_model /media/6T/yaqing/experiments/ontonotes5_gold/WIKI_NER_bert-base-uncased_false_all_avg_linear_128_true_false_bert_base_qa_linear/epoch_12/pytorch_model.bin'
     #model_path_arg='--path_to_model /media/6T/yaqing/experiments/ontonotes5_gold/WIKI_NER_bert-base-uncased_true_all_avg_linear_128_true_false_bert_base_qa_linear/epoch_8/pytorch_model.bin'
     model_path_arg='--path_to_model /media/6T/yaqing/experiments/ontonotes5_gold/bert-base-uncased_true_all_avg_linear_128_true_false_bert_base_qa_linear/epoch_5/pytorch_model.bin'
   fi
  if [ "${data}" != "WIKI_NER"  ]
  then
    echo $use_attention

    if [ "${use_attention}" = "true" ]
    then
      cand_enc_args="--use_attention ${cand_enc_args} --freeze_cand_enc --load_cand_enc_only --cand_enc_path ${data_path}/${embedding_choice}_base_seq_embedding_label.pt --cand_token_ids_path ${data_path}/wiki_tokenized_label.pt"
    else
      cand_enc_args="--freeze_cand_enc ${cand_enc_args} --load_cand_enc_only --cand_enc_path ${data_path}/${embedding_choice}_base_avg_embedding_label.pt --cand_token_ids_path ${data_path}/wiki_tokenized_label.pt"
    fi
  elif [ "${data}" = "WIKI_NER" ]
  then

    if [ "${use_attention}" = "true" ]
    then
      cand_enc_args="--use_attention --freeze_cand_enc --load_cand_enc_only --cand_enc_path ${data_path}/wiki_base_seq_embedding_label.pt --cand_token_ids_path ${data_path}/wiki_tokenized_label.pt"
    else
      cand_enc_args="--freeze_cand_enc --load_cand_enc_only --cand_enc_path ${data_path}/wiki_base_avg_embedding_label.pt --cand_token_ids_path ${data_path}/wiki_tokenized_label.pt"
    fi

  fi
  echo ${cand_enc_args}

  cmd="python src/biencoder/train_biencoder.py \
    --output_path ${model_dir} \
    ${model_path_arg} ${cand_enc_args} \
    --title_key entity \
    --data_path ${data_path} \
    --num_train_epochs 1000 \
    --learning_rate ${learning_rate} \
    --max_context_length ${context_length} \
    --max_cand_length 128 \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --bert_model ${model_ckpt} \
    --mention_scoring_method ${mention_scoring_method} \
    --eval_interval 20 \
    --last_epoch ${epoch} \
    --shuffle True  \
    --gpu_ids ${gpu_ids} \
    ${all_mention_args}  --get_losses ${dont_distribute_train_samples}"     #--data_parallel  ${distribute_train_samples_arg} --debug  #

  echo $cmd
  $cmd

fi

# --path_to_model ${model_path} \