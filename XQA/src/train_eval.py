import torch, os, logging, timeit, json
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from datetime import datetime, timezone, timedelta
from utils import load_and_cache_examples
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from mlqa_evaluation_v1 import mlqa_evaluate, evaluate_mlqa
from evaluate_v1_1 import evaluate_xquad
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)

logger = logging.getLogger(__name__)
logging.getLogger("transformers.data.metrics.squad_metrics").setLevel(logging.WARN)  # Reduce model loading logs

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(args, model, tokenizer):
    """ Train the model """
    exec_time = datetime.utcnow().astimezone(timezone(timedelta(hours=8))) \
        .strftime("%Y-%m-%d-%H-%M")
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=os.path.join('../logs', exec_time))
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate='train', output_examples=False)
    # train_dataset = load_and_cache_examples(args, tokenizer, evaluate='dev', context_lang='en', query_lang='en', output_examples=False)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio*t_total, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)
    best_f1 = 0
    best_em = 0
    results = {}
    en_valid_only = False
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
                "pos_label": batch[-3],
                "dep_graph_coo": batch[-2],
                "dep_graph_etype": batch[-1]
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            # if isinstance(model, torch.nn.DataParallel):
            #     inputs["return_tuple"] = True

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss_qa, loss_cf1, loss_cf2 = outputs[0], outputs[1], outputs[2]
            loss = loss_qa + args.loss_scale_1 * loss_cf1 + args.loss_scale_2 * loss_cf2
            # loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics

                tb_writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("train/loss", loss, global_step)
                tb_writer.add_scalar("train/loss_qa", loss_qa.mean(), global_step)
                tb_writer.add_scalar("train/loss_cf1", loss_cf1.mean(), global_step)
                tb_writer.add_scalar("train/loss_cf2", loss_cf2.mean(), global_step)
                logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and global_step > int(t_total/2):
                    dev_f1 = {}
                    dev_em = {}
                    for l in ['en', 'es', 'de', 'ar', 'hi', 'vi', 'zh']:
                        result = evaluate(args, model, tokenizer, prefix=global_step,
                                          set='dev', context_lang=l, query_lang=l)
                        dev_f1[l] = result['f1']
                        dev_em[l] = result['exact_match']
                    tb_writer.add_scalars("valid_f1", dev_f1, global_step)
                    tb_writer.add_scalars("valid_em", dev_em, global_step)

                    if en_valid_only:
                        now_f1 = dev_f1['en']
                        now_em = dev_em['en']
                    else:
                        f1_list = list(dev_f1.values())
                        em_list = list(dev_em.values())
                        now_f1 = sum(f1_list)/len(f1_list)
                        now_em = sum(em_list)/len(em_list)

                    if now_f1 > best_f1:
                        best_f1 = now_f1
                        best_em = now_em
                        logger.info(
                            "Now Best checkpoint and its dev em/f1 result: {}, {}/{}".format(global_step, round(best_em,2), round(best_f1,2)))
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "{}-checkpoint-best".format(exec_time))
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        dev_f1 = {}
        dev_em = {}
        for l in ['en', 'es', 'de', 'ar', 'hi', 'vi', 'zh']:
            result = evaluate(args, model, tokenizer, prefix=global_step,
                              set='dev', context_lang=l, query_lang=l)
            dev_f1[l] = result['f1']
            dev_em[l] = result['exact_match']
        tb_writer.add_scalars("valid_f1", dev_f1, global_step)
        tb_writer.add_scalars("valid_em", dev_em, global_step)

        if en_valid_only:
            now_f1 = dev_f1['en']
            now_em = dev_em['en']
        else:
            f1_list = list(dev_f1.values())
            em_list = list(dev_em.values())
            now_f1 = sum(f1_list) / len(f1_list)
            now_em = sum(em_list) / len(em_list)

        if now_f1 > best_f1:
            best_f1 = now_f1
            best_em = now_em
            logger.info(
                "Now Best checkpoint and its dev em/f1 result: {}, {}/{}".format(global_step, round(best_em, 2),
                                                                                 round(best_f1, 2)))
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-checkpoint-best".format(exec_time))
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step, exec_time


def evaluate(args, model, tokenizer, prefix="", set='dev', context_lang='en', query_lang='en', IS_XQUAD=False):
    dataset, examples, features = load_and_cache_examples(args,
                                                          tokenizer,
                                                          evaluate=set,
                                                          context_lang=context_lang,
                                                          query_lang=query_lang,
                                                          output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    # logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(dataset))
    # logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "pos_label": batch[-3],
                "dep_graph_coo": batch[-2],
                "dep_graph_etype": batch[-1]
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            # if isinstance(model, torch.nn.DataParallel):
            #     inputs["return_tuple"] = True
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output[0], output[1]
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    # logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    prefix = set+'_'+context_lang+'_'+query_lang
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    if set != 'xquad':
        gold_file = '{}-context-{}-question-{}.json'.format(set, context_lang, query_lang)
        dataset_file = os.path.join('../data', set, gold_file)
        with open(dataset_file) as f:
            dataset = json.load(f)
            dataset = dataset['data']
        results = evaluate_mlqa(dataset, predictions, context_lang)
    else:
        gold_file = 'xquad.{}.json'.format(context_lang)
        dataset_file = os.path.join('../data', set, gold_file)
        with open(dataset_file) as f:
            dataset = json.load(f)
            dataset = dataset['data']
        results = evaluate_xquad(dataset, predictions)

    return results
