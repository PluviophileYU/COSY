import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default='../data/',
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set.")

    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0.1, type=int, help="Linear warmup ratio over warmup_steps.")
    parser.add_argument("--loss_scale_1", default=0.1, type=float)
    parser.add_argument("--loss_scale_2", default=0.1, type=float)
    parser.add_argument("--gan_dropout_prob", default=0.1, type=float)
    parser.add_argument("--addtional_feature_size", default=100, type=int)

    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    # parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    # parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()
    return args

Pos2idx = {
    "ADJ": 1,
    "ADV": 2,
    "INTJ": 3,
    "NOUN": 4,
    "PROPN": 5,
    "VERB": 6,
    "ADP": 7,
    "AUX": 8,
    "CCONJ": 9,
    "DET": 10,
    "NUM": 11,
    "PART": 12,
    "PRON": 13,
    "SCONJ": 14,
    "PUNCT": 15,
    "SYM": 16,
    "X": 17
}

Rel2idx = {
    'pad': 0,
    'self_loop': 1,
    'acl': 2,
    'advcl': 3,
    'advmod': 4,
    'amod': 5,
    'appos': 6,
    'aux': 7,
    'case': 8,
    'cc': 9,
    'ccomp': 10,
    'clf': 11,
    'compound': 12,
    'conj': 13,
    'cop': 14,
    'csubj': 15,
    'dep': 16,
    'det': 17,
    'discourse': 18,
    'dislocated': 19,
    'expl': 20,
    'fixed': 21,
    'flat': 22,
    'goeswith': 23,
    'iobj': 24,
    'list': 25,
    'mark': 26,
    'nmod': 27,
    'nsubj': 28,
    'nummod': 29,
    'obj': 30,
    'obl': 31,
    'orphan': 32,
    'parataxis': 33,
    'punct': 34,
    'reparandum': 35,
    'root': 36,
    'vocative': 37,
    'xcomp': 38
}

#
preprocessing_data = [('dev','en'),
                    #   ('dev','es'),
                    #   ('dev','de'),
                    #   ('dev','ar'),
                    #   ('dev','hi'),
                    #   ('dev','vi'),
                    #   ('dev','zh'),
                    # ('xquad', 'en'),
                    # ('xquad', 'ar'),
                    # ('xquad', 'de'),
                    # ('xquad', 'el'),
                    # ('xquad', 'es'),
                    # ('xquad', 'hi'),
                    # ('xquad', 'ru'),
                    # ('xquad', 'tr'),
                    # ('xquad', 'vi'),
                    ('xquad', 'zh'),
]

# preprocessing_data = [('test','en'),
#                       ('test','es'),
#                       ('test','de'),
#                       ('test','ar'),
#                       ('test','hi'),
#                       ('test','vi'),
#                       ('test','zh'),
# ]

# preprocessing_data = [('dev','en'),
#                       ('test','en'),
#                       ('test','es'),
#                       ('test','de'),
#                       ('test','ar'),
#                       ('test','hi'),
#                       ('test','vi'),
#                       ('test','zh'),
#                       ('dev','es'),
#                       ('dev','de'),
#                       ('dev','ar'),
#                       ('dev','hi'),
#                       ('dev','vi'),
#                       ('dev','zh'),
#                     ('xquad', 'en'),
#                     ('xquad', 'ar'),
#                     ('xquad', 'de'),
#                     ('xquad', 'el'),
#                     ('xquad', 'es'),
#                     ('xquad', 'hi'),
#                     ('xquad', 'ru'),
#                     ('xquad', 'tr'),
#                     ('xquad', 'vi'),
#                     ('xquad', 'zh'),
# ]

large_mismatch = []
for i in range(15,50):
    item = []
    for j in range(1,i):
        item.append((j,i-j,abs(2*j-i)))
    item = sorted(item, key=lambda tup: tup[2])
    item = [(k[0],k[1]) for k in item]
    large_mismatch.extend(item)

mismatch = [(1,1),
            (2,1), (1,2),
            (2,2),
            (3,1), (1,3),
            (4,1), (1,4),
            (2,3), (3,2),
            (4,2), (2,4),
            (3,3),
            (1,5), (5,1),
            (4,3), (3,4),
            (5,2), (2,5),
            (1,6), (6,1),
            (4,4),
            (1,7), (7,1),
            (2,6), (6,2),
            (3,5), (5,3),
            (4,5), (5,4),
            (3,6), (6,3),
            (2,7), (7,2),
            (1,8), (8,1),
            (5,5),
            (4,6), (6,4),
            (3,7), (7,3),
            (2,8), (8,2),
            (1,9), (9,1),
            (5,6), (6,5),
            (4,7), (7,4),
            (3,8), (8,3),
            (2,9), (9,2),
            (1,10), (10,1),
            (6,6),
            (5,7), (7,5),
            (4,8), (8,4),
            (3,9), (9,3),
            (2,10), (10,2),
            (1,11), (11,1),
            (6,7), (7,6),
            (5,8), (8,5),
            (4,9), (9,4),
            (3,10), (10,3),
            (2,11), (11,2),
            (1,12), (12,1),
            (7,7),
            (6,8), (8,6),
            (5,9), (9,5),
            (4,10), (10,4),
            (3,11), (11,3),
            (2,12), (12,2),
            (1,13), (13,1),
            ]
mismatch.extend(large_mismatch)