# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import argparse
import faiss
import pickle
import torch
import json
import sys
import io
import random
import time
import traceback
import numpy as np
from scipy.special import softmax, expit

import torch.nn.functional as F

from multiprocessing.pool import ThreadPool

from tqdm import tqdm, trange
from collections import OrderedDict

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.optimization import WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer


from src.biencoder.biencoder import BiEncoderRanker
from src.vcg_utils.measures import entity_linking_tp_with_overlap
import logging

import src.candidate_ranking.utils as utils
from src.biencoder.data_process import process_mention_data
from src.common.optimizer import get_bert_optimizer
from src.common.params import SpanParser
from src.index.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer, DenseIVFFlatIndexer
import pdb

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
def transfer_to_iob(gold_mb,gold_label, label_map):
    label_list= ['O'] * 128
    for i, mb in enumerate(gold_mb):
        start = mb[0]
        end = mb[1]+1
        label = label_map[gold_label[i]]
        for j in range(start, end):
            if j == start:
                tag='B-'
            else:
                tag = 'I-'
            tag += label
            label_list[j]=tag
    return label_list





logger = None
np.random.seed(1234)  # reproducible for FAISS indexer

def evaluate(
    reranker, eval_dataloader, params, device, logger,
    cand_encs=None, faiss_index=None,
    get_losses=False, cand_attention_mask=None, threshold=1
):

    reranker.model.eval()
    if params["silent"]:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}
    # import pdb
    # pdb.set_trace()

    eval_num_correct = 0.0
    eval_num_span_correct = 0.0
    eval_num_p = 0.0
    eval_num_g = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0
    overall_loss = 0.0
    y_true_list = []
    y_pred_list = []

    if cand_encs is not None and not params["freeze_cand_enc"]:
        cand_encs = cand_encs.to(device)
        cand_attention_mask.to(device)
    cand_encs_cpu = cand_encs.cpu()

    label_map = json.load(open(os.path.join(params['data_path'], 'label_map.json')))
    label_map = {label_map[name]:name for name in label_map}

    test_file = open(os.path.join(params['data_path'], 'test.jsonl'))
    test_data = []
    label_mask = []
    for l in test_file:
        test_data.append(json.loads(l)['text'])
        label_mask.append(json.loads(l)['start_label_mask'])
    index = 0



    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]	
        candidate_input = batch[1]
        # (bs, num_actual_spans)
        label_ids = batch[2].cpu().numpy() if params["freeze_cand_enc"] else None
        if params["debug"] and label_ids is not None:
            label_ids[label_ids > 199] = 199
        mention_idx = batch[5]
        mention_idx_mask = batch[7]
        start_label_mask = batch[8]
        end_label_mask = batch[9]

        
        with torch.no_grad():
            # evaluate with joint mention detection
            if params["freeze_cand_enc"]:
                if params['entity_inference']:
                    context_outs = reranker.encode_context(
                        context_input,
                        num_cand_mentions=50,
                        topK_threshold=-10000,
                        start_label_mask=start_label_mask,
                        end_label_mask=end_label_mask,
                        gold_mention_bounds=mention_idx, gold_mention_bounds_mask=mention_idx_mask
                    )
                else:
                    if params['zero_shot']:
                        context_outs = reranker.encode_context(
                            context_input,
                            num_cand_mentions=30,
                            topK_threshold=0,
                            start_label_mask=start_label_mask,
                            end_label_mask=end_label_mask,

                        )
                    else:
                        context_outs = reranker.encode_context(
                            context_input,
                            num_cand_mentions=30,
                            topK_threshold=-1,
                            start_label_mask=start_label_mask,
                            end_label_mask=end_label_mask,

                        )

                mention_idx = batch[5].cpu().numpy()
                mention_idx_mask = batch[7].cpu().numpy()
                if faiss_index is not None:
                    embedding_context = context_outs['mention_reps'].cpu().numpy()
                    pred_mention_mask = context_outs['mention_masks'].cpu().numpy()
                else:
                    embedding_context = context_outs['mention_reps'].cpu()
                    pred_mention_mask = context_outs['mention_masks'].cpu()
                chosen_mention_bounds = context_outs['mention_bounds'].cpu().numpy()
                pred_mention_mask = pred_mention_mask.bool()


                #pred_mention_mask = mention_idx_mask
                embedding_ctxt = embedding_context[pred_mention_mask]
                pred_mention_mask = pred_mention_mask.numpy()


                if params['use_attention']:
                    if embedding_ctxt.size(0) != 0:
                        top_cand_logits_shape = reranker.get_scores(embedding_ctxt, cand_encs, cand_attention_mask).cpu()
                    else:
                        top_cand_logits_shape = embedding_ctxt.cpu().mm(cand_encs.cpu().mean(1).t())
                else:
                    top_cand_logits_shape = embedding_ctxt.mm(cand_encs.t()).cpu()

                top_cand_logits_shape, top_cand_indices_shape = torch.sort(top_cand_logits_shape, descending=True, dim=-1)

                top_cand_logits = np.zeros((pred_mention_mask.shape[0], pred_mention_mask.shape[1], top_cand_indices_shape.size(-1)), dtype=np.float)
                top_cand_indices = np.zeros_like(pred_mention_mask, dtype=np.int)
                top_cand_logits[pred_mention_mask] = top_cand_logits_shape
                if len(top_cand_indices_shape.size()) == 1:
                    top_cand_indices_shape = top_cand_indices_shape.unsqueeze(0)

                top_cand_indices[pred_mention_mask] = top_cand_indices_shape[:,0]



                type_prob = top_cand_logits[:,:,0]
                # import pdb
                # pdb.set_trace()
                if not params['entity_inference']:

                    scores = (np.log(softmax(top_cand_logits, -1)) + torch.sigmoid(
                    context_outs['mention_logits'].unsqueeze(-1)).log().cpu().numpy())[:, :, 0]


                tmp_num_correct = 0.0
                tmp_num_p = 0.0
                tmp_num_g = 0.0
                tmp_num_span_correct = 0.0



                for i, ex in enumerate(top_cand_indices):
                    gold_mb = mention_idx[i][mention_idx_mask[i]]
                    gold_label_ids = label_ids[i][mention_idx_mask[i]]

                    y_true = transfer_to_iob(gold_mb, gold_label_ids, label_map)
                    y_true_list.append(y_true)
                    #overall_score_mask = scores[i][pred_mention_mask[i]] > -2.5
                    #overall_score_mask = scores[i][pred_mention_mask[i]] > -5
                    if params['entity_inference']:
                        unknown_score_mask = (type_prob[i][pred_mention_mask[i]] > -100000) #& overall_score_mask
                    else:
                        if not params['zero_shot']:
                            unknown_score_mask = (type_prob[i][pred_mention_mask[i]] > 0)

                    if params['entity_inference']:
                        overall_score_mask = unknown_score_mask
                    else:
                        overall_score_mask = scores[i][pred_mention_mask[i]] > -threshold
                        overall_score_mask = (type_prob[i][pred_mention_mask[i]] > 0)

                    try:
                        pred_mb = chosen_mention_bounds[i][pred_mention_mask[i]][overall_score_mask]
                    except:
                        import  pdb
                        pdb.set_trace()
                    pred_label_ids = ex[pred_mention_mask[i]][overall_score_mask]
                    y_pred = transfer_to_iob(pred_mb, pred_label_ids , label_map)
                    y_pred_list.append(y_pred)
                    try:
                        gold_triples = [(str(gold_label_ids[j]), gold_mb[j][0], gold_mb[j][1]) for j in range(len(gold_mb))]
                    except:
                        import pdb
                        pdb.set_trace()
                    pred_triples = [(str(pred_label_ids[j]), pred_mb[j][0], pred_mb[j][1]) for j in range(len(pred_mb))]
                    try:
                        num_overlap_weak, num_overlap_strong, num_overlap_span = entity_linking_tp_with_overlap(gold_triples, pred_triples)
                    except:
                        import pdb
                        pdb.set_trace()
                    tmp_num_correct += num_overlap_strong
                    tmp_num_span_correct += num_overlap_span
                    tmp_num_p += float(len(pred_triples))
                    tmp_num_g += float(len(gold_triples))
                text_encs = embedding_context
            else:
                loss, logits, mention_logits, mention_bounds = reranker(
                    context_input, candidate_input,
                    cand_encs=cand_encs,
                    gold_mention_bounds=batch[-2],
                    gold_mention_bounds_mask=batch[-1],
                    return_loss=True,
                    start_label_mask=start_label_mask,
                    end_label_mask=end_label_mask
                )
                logits = logits.cpu().numpy()
                # Using in-batch negatives, the label ids are diagonal
                label_ids = torch.LongTensor(torch.arange(logits.shape[0]))
                label_ids = label_ids.cpu().numpy()
                tmp_num_correct = utils.accuracy(logits, label_ids)
                tmp_num_p = len(batch[-2][batch[-1]])
                tmp_num_g = len(batch[-2][batch[-1]])

                overall_loss += loss

        eval_num_correct += tmp_num_correct
        eval_num_p += tmp_num_p
        eval_num_g += tmp_num_g
        eval_num_span_correct += tmp_num_span_correct

        nb_eval_steps += 1



    if cand_encs is not None:
        cand_encs = cand_encs.to("cpu")
        torch.cuda.empty_cache()

    if nb_eval_steps > 0 and overall_loss > 0:
        normalized_overall_loss = overall_loss / nb_eval_steps
        logger.info("Overall loss: %.5f" % normalized_overall_loss)
    if eval_num_p > 0:
        normalized_eval_p = eval_num_correct / eval_num_p
        normalized_eval_span_p = eval_num_span_correct / eval_num_p
    else:
        normalized_eval_p = 0.0
        normalized_eval_span_p = 0.0
    if eval_num_g > 0:
        normalized_eval_r = eval_num_correct / eval_num_g
        normalized_eval_span_r = eval_num_span_correct / eval_num_g
    else:
        normalized_eval_r = 0.0
        normalized_eval_span_r = 0.0

    logger.info("Precision: %.5f" % normalized_eval_p)
    logger.info("Recall: %.5f" % normalized_eval_r)
    logger.info("Span Precision: %.5f" % normalized_eval_span_p)
    logger.info("Span Recall: %.5f" % normalized_eval_span_r)
    if normalized_eval_p + normalized_eval_r == 0:
        f1 = 0
    else:
        f1 = 2 * normalized_eval_p * normalized_eval_r / (normalized_eval_p + normalized_eval_r)
    if normalized_eval_span_p + normalized_eval_span_r == 0:
        span_f1 = 0
    else:
        span_f1 = 2 * normalized_eval_span_p * normalized_eval_span_r / (normalized_eval_span_p + normalized_eval_span_r)
    logger.info("F1: %.5f" % f1)
    logger.info("Span F1: %.5f" % span_f1)
    results["normalized_f1"] = f1
    logger.info("threshold: %.2f" % threshold)

    test_pred = open(os.path.join(params['data_path'], 'test_pred.txt'), 'w')
    # import pdb
    #
    # pdb.set_trace()

    for i, line in enumerate(test_data):
        line = line.split()
        labels = y_true_list[i]
        label_idx = 0
        for j,word in enumerate(line):
            if label_idx >= len(label_mask[i]):
                break
            while label_mask[i][label_idx] == 0:
                label_idx += 1
                if label_idx >= len(label_mask[i]):
                    break

            if label_idx >= len(labels)-1:
                continue

            label = labels[label_idx+1]
            test_pred.write(word +' '+label+' '+y_pred_list[i][label_idx+1]+'\n')
            label_idx += 1
        test_pred.write('\n')
    test_pred.close()








    print(classification_report(y_true_list, y_pred_list, digits=3))




    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps,
    # )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler



def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    #pdb.set_trace()
    if len(params['gpu_ids']) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in params['gpu_ids'].split(','))
        params['data_parallel'] = True
        params["train_batch_size"] = params["train_batch_size"] * len(params['gpu_ids'].split(','))
        params["eval_batch_size"] = params["eval_batch_size"] * len(params['gpu_ids'].split(','))
    elif int(params['gpu_ids']) != -1:
        params['data_parallel'] = False
        device_id=int(params['gpu_ids'])
        torch.cuda.set_device(device_id)
    else:
        params['data_parallel'] = True
        params["train_batch_size"] = params["train_batch_size"] * torch.cuda.device_count()
        params["eval_batch_size"] = params["eval_batch_size"] * torch.cuda.device_count()



    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    device = reranker.device
    n_gpu = reranker.n_gpu

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )



    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    params["train_batch_size"] = (
        params["train_batch_size"] // params["gradient_accumulation_steps"]
    )
    train_batch_size = params["train_batch_size"]
    eval_batch_size = params["eval_batch_size"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    # Load train data
    train_samples = utils.read_dataset("train", params["data_path"])
    test_samples = utils.read_dataset("test", params["data_path"])
    logger.info("Read %d train samples." % len(train_samples))
    logger.info("Finished reading all train samples")

    # Load eval data
    try:
        valid_samples = utils.read_dataset("valid", params["data_path"])
    except FileNotFoundError:
        valid_samples = utils.read_dataset("dev", params["data_path"])
    # MUST BE DIVISBLE BY n_gpus
    if len(valid_samples) > 2048:
        valid_subset = 2048
    else:
        valid_subset = len(valid_samples) - len(valid_samples) % torch.cuda.device_count()


    logger.info("Read %d valid samples, choosing %d subset" % (len(valid_samples), valid_subset))

    valid_data, valid_tensor_data, extra_ret_values = process_mention_data(
        samples=valid_samples[:valid_subset],  # use subset of valid data valid_samples[:valid_subset]
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        context_key=params["context_key"],
        title_key=params["title_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not args.no_mention_bounds),
        candidate_token_ids=None,
        params=params,
    )
    candidate_token_ids = extra_ret_values["candidate_token_ids"]
    valid_tensor_data = TensorDataset(*valid_tensor_data)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=eval_batch_size
    )



    test_data, test_tensor_data, extra_ret_values = process_mention_data(
        samples=test_samples,  # use subset of valid data valid_samples[:valid_subset]
        tokenizer=tokenizer,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        context_key=params["context_key"],
        title_key=params["title_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
        add_mention_bounds=(not args.no_mention_bounds),
        candidate_token_ids=None,
        params=params,
    )


    test_tensor_data = TensorDataset(*test_tensor_data)
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(
        test_tensor_data, sampler=test_sampler, batch_size=eval_batch_size
    )

    # load candidate encodings

    cand_encs = None
    cand_encs_index = None
    cand_attention_mask = None

    if params["freeze_cand_enc"]:
        all_cand_encs = torch.load(params['cand_enc_path'])
        if params['use_attention']:
            cand_encs = all_cand_encs[0][0]
            cand_attention_mask = all_cand_encs[1]
        else:
            cand_encs = all_cand_encs

        logger.info("Loaded saved entity encodings")
        if params["debug"]:
            cand_encs = cand_encs[:200]


        # build FAISS index
        if len(cand_encs) > 10000:
            cand_encs_index = DenseHNSWFlatIndexer(1)
            cand_encs_index.deserialize_from(params['index_path'])
            logger.info("Loaded FAISS index on entity encodings")
            num_neighbors = 10
        else:
            cand_encs_index = None


    time_start = time.time()
    best_f1 = 0

    utils.write_to_file(
        os.path.join(model_output_path, "training_params.txt"), str(params)
    )

    logger.info("Starting training")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False)
    )

    num_train_epochs = params["num_train_epochs"]

    if params["dont_distribute_train_samples"]:
        num_samples_per_batch = len(train_samples)

        train_data, train_tensor_data_tuple, extra_ret_values = process_mention_data(
            samples=train_samples,
            tokenizer=tokenizer,
            max_context_length=params["max_context_length"],
            max_cand_length=params["max_cand_length"],
            context_key=params["context_key"],
            title_key=params["title_key"],
            silent=params["silent"],
            logger=logger,
            debug=params["debug"],
            add_mention_bounds=(not args.no_mention_bounds),
            candidate_token_ids=candidate_token_ids,
            params=params,
        )
        logger.info("Finished preparing training data")
    else:
        num_samples_per_batch = len(train_samples) // num_train_epochs




    trainer_path = params.get("path_to_trainer_state", None)
    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(
        params, optimizer, num_samples_per_batch,
        logger
    )
    if trainer_path is not None and os.path.exists(trainer_path):
        training_state = torch.load(trainer_path)
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])
        logger.info("Loaded saved training state")

    model.train()

    best_epoch_idx = -1
    best_score = -1
    global_step = 0
    logger.info("Num samples per batch : %d" % num_samples_per_batch)
    for epoch_idx in trange(params["last_epoch"] + 1, int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if not params["dont_distribute_train_samples"]:
            start_idx = epoch_idx * num_samples_per_batch
            end_idx = (epoch_idx + 1) * num_samples_per_batch

            train_data, train_tensor_data_tuple, extra_ret_values = process_mention_data(
                samples=train_samples[start_idx:end_idx],
                tokenizer=tokenizer,
                max_context_length=params["max_context_length"],
                max_cand_length=params["max_cand_length"],
                context_key=params["context_key"],
                title_key=params["title_key"],
                silent=params["silent"],
                logger=logger,
                debug=params["debug"],
                add_mention_bounds=(not args.no_mention_bounds),
                candidate_token_ids=candidate_token_ids,
                params=params,
            )
            logger.info("Finished preparing training data for epoch {}: {} samples".format(epoch_idx, len(train_tensor_data_tuple[0])))
    
        batch_train_tensor_data = TensorDataset(
            *list(train_tensor_data_tuple)
        )
        if params["shuffle"]:
            train_sampler = RandomSampler(batch_train_tensor_data)
        else:
            train_sampler = SequentialSampler(batch_train_tensor_data)

        train_dataloader = DataLoader(
            batch_train_tensor_data, sampler=train_sampler, batch_size=train_batch_size
        )

        # # save_k_shot_ctxt_embeding(
        # #     reranker, train_dataloader, params,
        # #     cand_encs=cand_encs, device=device,
        # #     logger=logger, faiss_index=cand_encs_index,
        # #     cand_attention_mask=cand_attention_mask
        # )



        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")



        for step, batch in enumerate(iter_):
            global_step += 1
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0]	
            candidate_input = batch[1]
            label_ids = batch[2] if params["freeze_cand_enc"] else None
            start_label = batch[3]
            end_label = batch[4]
            mention_idxs = batch[5]
            mention_idx_mask = batch[6]
            type_idx_mask = batch[7]
            start_label_mask = batch[8]
            end_label_mask = batch[9]
            #pdb.set_trace()
            if params["debug"] and label_ids is not None:
                label_ids[label_ids > 199] = 199

            cand_encs_input = None
            label_input = None
            mention_reps_input = None
            mention_logits = None
            mention_bounds = None
            hard_negs_mask = None
            pos_cand_encs_input = cand_encs[label_ids.to("cpu")]

            #pos_cand_encs_input[~mention_idx_mask] = 0
            pos_cand_encs_input[~type_idx_mask] = 0

            if params['use_attention']:
                pos_cand_attention_mask = cand_attention_mask[label_ids.to("cpu")]
                pos_cand_attention_mask[~type_idx_mask] =  0


            # (bs, num_spans, embed_size)



            # mention_reps: (bs, max_num_spans, embed_size) -> masked_mention_reps: (all_pred_mentions_batch, embed_size)

            #pdb.set_trace()

            context_outs = reranker.encode_context(
                context_input, gold_mention_bounds=mention_idxs,
                gold_mention_bounds_mask=mention_idx_mask,
                get_mention_scores=True,
            )
            #pdb.set_trace()
            mention_logits = context_outs['all_mention_logits']
            start_mention_logits = context_outs['all_mention_start_logits']
            end_mention_logits = context_outs['all_mention_end_logits']
            mention_bounds = context_outs['all_mention_bounds']
            mention_reps = context_outs['mention_reps']
            masked_mention_reps = mention_reps[context_outs['mention_masks']]

            #pdb.set_trace()

            assert cand_encs is not None and label_ids is not None  # due to params["freeze_cand_enc"] being set

            if params["adversarial_training"]:
                '''
                GET CLOSEST N CANDIDATES (AND APPROPRIATE LABELS)
                               '''

                # neg_cand_encs_input_idxs: (all_pred_mentions_batch, num_negatives)
                _, neg_cand_encs_input_idxs = cand_encs_index.search_knn(masked_mention_reps.detach().cpu().numpy(),
                                                                         num_neighbors)
                neg_cand_encs_input_idxs = torch.from_numpy(neg_cand_encs_input_idxs)

            else:
                valid_mention_num = mention_idx_mask.sum()
                neg_cand_encs_input_idxs = torch.arange(cand_encs.size(0)).unsqueeze(0).repeat(
                    valid_mention_num, 1)
                # neg_num = min(10, cand_encs.size(0))
                # neg_cand_encs_input_idxs = torch.randint(0, cand_encs.size(0), (valid_mention_num, neg_num))
                # import pdb
                # pdb.set_trace()


                # set "correct" closest entities to -1
                # masked_label_ids: (all_pred_mentions_batch)
                masked_label_ids = label_ids[mention_idx_mask]

                #pdb.set_trace()


                neg_cand_encs_input_idxs[neg_cand_encs_input_idxs - masked_label_ids.to("cpu").unsqueeze(-1) == 0] = -1

                # reshape back tensor (extract num_spans dimension)
                # (bs, num_spans, num_negatives)
                neg_cand_encs_input_idxs_reconstruct = torch.zeros(label_ids.size(0), label_ids.size(1),
                                                                   neg_cand_encs_input_idxs.size(-1),
                                                                   dtype=neg_cand_encs_input_idxs.dtype)
                neg_cand_encs_input_idxs_reconstruct[mention_idx_mask] = neg_cand_encs_input_idxs
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs_reconstruct

                # create neg_example_idx (corresponding example (in batch) for each negative)
                # neg_example_idx: (bs * num_negatives)
                neg_example_idx = torch.arange(neg_cand_encs_input_idxs.size(0)).unsqueeze(-1)
                neg_example_idx = neg_example_idx.expand(neg_cand_encs_input_idxs.size(0), neg_cand_encs_input_idxs.size(2))
                neg_example_idx = neg_example_idx.flatten()

                # flatten and filter -1 (i.e. any correct/positive entities)
                # neg_cand_encs_input_idxs: (bs * num_negatives, num_spans)
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs.permute(0, 2, 1)
                neg_cand_encs_input_idxs = neg_cand_encs_input_idxs.reshape(-1, neg_cand_encs_input_idxs.size(-1))
                # mask invalid negatives (actually the positive example)
                # (bs * num_negatives)

                #pdb.set_trace()


                # (bs * num_negatives - invalid_negs, num_spans, embed_size)
                neg_mention_idx_mask = mention_idx_mask[neg_example_idx]
                neg_mention_idx_mask = (~(~neg_mention_idx_mask.cpu() | (neg_cand_encs_input_idxs == -1))).to(device)

                neg_cand_encs_input = cand_encs[neg_cand_encs_input_idxs]

                # (bs * num_negatives - invalid_negs, num_spans, embed_size)
                #neg_mention_idx_mask = mention_idx_mask[neg_example_idx]
                neg_cand_encs_input[~neg_mention_idx_mask] = 0

                if params['use_attention']:
                    neg_cand_attention_mask = cand_attention_mask[neg_cand_encs_input_idxs]
                    neg_cand_attention_mask[~neg_mention_idx_mask] = 0


                # create input tensors (concat [pos examples, neg examples])
                if mention_reps is not None:
                    # (bs + bs * num_negatives, num_spans, embed_size)
                    mention_reps_input = torch.cat([
                        mention_reps, mention_reps[neg_example_idx.to(device)],
                    ])
                    assert mention_reps.size(0) == pos_cand_encs_input.size(0)

                # (bs + bs * num_negatives, num_spans)
                label_input = torch.cat([
                    torch.ones(pos_cand_encs_input.size(0), pos_cand_encs_input.size(1), dtype=label_ids.dtype),
                    torch.zeros(neg_cand_encs_input.size(0), neg_cand_encs_input.size(1), dtype=label_ids.dtype),
                ]).to(device)
                # (bs + bs * num_negatives, num_spans, embed_size)
                cand_encs_input = torch.cat([
                    pos_cand_encs_input, neg_cand_encs_input,
                ]).to(device)

                hard_negs_mask = torch.cat([type_idx_mask, neg_mention_idx_mask])
                all_cand_attention_mask = None
                if params['use_attention']:
                    all_cand_attention_mask = torch.cat([pos_cand_attention_mask, neg_cand_attention_mask]).to(device)
                    # else:
                    #     cand_encs_input
                #pdb.set_trace()

        


            loss, _, _, _ = reranker(
                context_input, candidate_input,
                cand_encs=cand_encs_input, text_encs=mention_reps_input,
                mention_logits=mention_logits, mention_bounds=mention_bounds,
                label_input=label_input, gold_mention_bounds=mention_idxs,
                gold_mention_bounds_mask=mention_idx_mask,
                hard_negs_mask=hard_negs_mask,
                return_loss=True,
                start_mention_logits=start_mention_logits,
                start_label=start_label,
                end_mention_logits=end_mention_logits,
                end_label=end_label, all_cand_attention_mask=all_cand_attention_mask,
            )

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()





            if (global_step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                loss = None  # for GPU mem management
                mention_reps = None
                mention_reps_input = None
                label_input = None
                cand_encs_input = None

                results = evaluate(
                    reranker, test_dataloader, params,
                    cand_encs=cand_encs, device=device,
                    logger=logger, faiss_index=cand_encs_index,
                    get_losses=params["get_losses"],
                    cand_attention_mask=cand_attention_mask
                )
                model.train()
                logger.info("\n")
                if results["normalized_f1"] > best_f1:
                    best_f1 = results["normalized_f1"]

                    logger.info("***** Saving fine - tuned model *****")
                    epoch_output_folder_path = os.path.join(
                        model_output_path, "epoch_{}".format(epoch_idx)
                    )
                    utils.save_model(model, tokenizer, epoch_output_folder_path)
                    torch.save({
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, os.path.join(epoch_output_folder_path, "training_state.th"))

        # logger.info("***** Saving fine - tuned model *****")
        # epoch_output_folder_path = os.path.join(
        #     model_output_path, "epoch_{}".format(epoch_idx)
        # )
        # utils.save_model(model, tokenizer, epoch_output_folder_path)
        # torch.save({
        #     "optimizer": optimizer.state_dict(),
        #     "scheduler": scheduler.state_dict(),
        # }, os.path.join(epoch_output_folder_path, "training_state.th"))

        # output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        # logger.info("Valid data evaluation")
        # results = evaluate(
        #     reranker, test_dataloader, params,
        #     cand_encs=cand_encs, device=device,
        #     logger=logger, faiss_index=cand_encs_index,
        #     get_losses=params["get_losses"],
        #     cand_attention_mask=cand_attention_mask
        # )
        if  results is not None and results["normalized_f1"] > best_f1:
            best_f1 = results["normalized_f1"]

            logger.info("***** Saving fine - tuned model *****")
            epoch_output_folder_path = os.path.join(
                model_output_path, "epoch_{}".format(epoch_idx)
            )
            utils.save_model(model, tokenizer, epoch_output_folder_path)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(epoch_output_folder_path, "training_state.th"))

        #logger.info("Train data evaluation")

        # results = evaluate(
        #     reranker, train_dataloader, params,
        #     cand_encs=cand_encs, device=device,
        #     logger=logger, faiss_index=cand_encs_index,
        #     get_losses=params["get_losses"],
        #     cand_attention_mask=cand_attention_mask
        # )

            ls = [best_score, results["normalized_f1"]]
            li = [best_epoch_idx, epoch_idx]

            best_score = ls[np.argmax(ls)]
            best_epoch_idx = li[np.argmax(ls)]
            logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(
        model_output_path, "epoch_{}".format(best_epoch_idx)
    )
    utils.save_model(reranker.model, tokenizer, model_output_path)

    if params["evaluate"]:
        params["path_to_model"] = model_output_path
        evaluate(params, cand_encs=cand_encs, logger=logger, faiss_index=cand_encs_index)


if __name__ == "__main__":
    parser = SpanParser(add_model_args=True)

    parser.add_training_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)
