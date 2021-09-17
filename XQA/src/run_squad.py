# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import glob
import logging
import os
import torch
from args import get_args, preprocessing_data
from utils import load_and_cache_examples
from train_eval import train, evaluate, set_seed
from prettytable import PrettyTable
from model import *
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

LANGS = ['en', 'es', 'de', 'ar', 'hi', 'vi', 'zh']
XQ_LANGS = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ru', 'tr', 'vi', 'zh']
# LANGS = ['en', 'es', 'de', 'hi', 'vi', 'zh']

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    args = get_args()
    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        print(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce model loading logs
    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.addtional_feature_size = args.addtional_feature_size
    config.gan_dropout_prob = args.gan_dropout_prob

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.model_type == 'bert':
        QAModel = mBertForQuestionAnswering_dep_beta_v3
    model = QAModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Data preprocessing with the dev and test data firstly (prevent fp16 issue when facing Stanza)
    # for set_name, lang in preprocessing_data:
    #     logger.info("Now process dev/test/xquad data: {}/{}".format(set_name, lang))
    #     dataset, examples, features = load_and_cache_examples(args,
    #                                                       tokenizer,
    #                                                       evaluate=set_name,
    #                                                       context_lang=lang,
    #                                                       query_lang=lang,
    #                                                       output_examples=True)


    # Training
    if args.do_train:
        # train_dataset = load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False)
        global_step, tr_loss, time_stamp = train(args, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            time_stamp = '12-02-11-14'
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = [i for i in checkpoints if time_stamp in i]

        logger.info("Evaluate the following checkpoints for dev: %s", checkpoints)

        best_f1 = 0
        best_em = 0
        best_ckpt = checkpoints[0]
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            logger.info("Load the checkpoint: {}".format(checkpoint))
            model = QAModel.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step, set='dev')

            if result['f1'] > best_f1:
                best_f1 = result['f1']
                best_em = result['exact_match']
                best_ckpt = checkpoint
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Dev Results: {}".format(results))
    logger.info("Best checkpoint and its dev em/f1 result: {}, {}/{}".format(best_ckpt, best_em, best_f1))
    if args.do_test and args.local_rank in [-1, 0]:
        model = QAModel.from_pretrained(best_ckpt)  # , force_download=True)
        model.to(args.device)
        logger.info("Evaluate on MLQA dataset!")
        mean_em = 0
        mean_f1 = 0
        table = PrettyTable()
        table.add_column(' ', ['EM', 'F1'])
        for lang in LANGS:
            result = evaluate(args, model, tokenizer, set='test', context_lang=lang, query_lang=lang, prefix=global_step)
            table.add_column(lang, [round(result['exact_match'], 2), round(result['f1'], 2)])
            # logger.info("Test Results for {}-{}: {}".format(lang,lang,result))
            mean_em += result['exact_match']
            mean_f1 += result['f1']
        mean_em = mean_em/len(LANGS)
        mean_f1 = mean_f1/len(LANGS)
        table.add_column('Avg', [round(mean_em, 2), round(mean_f1, 2)])
        print(table)


        logger.info("Evaluate on XQUAD dataset!")
        mean_em = 0
        mean_f1 = 0
        table = PrettyTable()
        table.add_column(' ', ['EM', 'F1'])
        for lang in XQ_LANGS:
            result = evaluate(args, model, tokenizer, set='xquad', context_lang=lang, query_lang=lang, prefix=global_step)
            table.add_column(lang, [round(result['exact_match'], 2), round(result['f1'], 2)])
            # logger.info("Test Results for {}-{}: {}".format(lang, lang, result))
            mean_em += result['exact_match']
            mean_f1 += result['f1']
        mean_em = mean_em / len(XQ_LANGS)
        mean_f1 = mean_f1 / len(XQ_LANGS)
        table.add_column('Avg', [round(mean_em, 2), round(mean_f1, 2)])
        print(table)

    return results


if __name__ == "__main__":
    main()
