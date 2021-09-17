import copy
import logging
import os
import re
from fuzzywuzzy import fuzz
from args import Pos2idx, Rel2idx, mismatch
import torch
from torch.utils.data import TensorDataset
import numpy as np
from tqdm import tqdm
logger = logging.getLogger(__name__)
from transformers.data.processors.squad import _improve_answer_span, \
    _new_check_is_max_context, \
    whitespace_tokenize, \
    SquadFeatures, \
    SquadV1Processor
import stanza
import numpy as np
from nltk.tokenize import sent_tokenize

def load_and_cache_examples(args, tokenizer, context_lang='en', query_lang='en', evaluate='train', output_examples=False, sent_level=True):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load stanford tagger
    assert context_lang == query_lang
    stanza_nlp = stanza.Pipeline(context_lang, logging_level='WARN', tokenize_pretokenized=True)
    stanza_nlp_ = stanza.Pipeline(context_lang, logging_level='WARN')
    sspliter = stanza.Pipeline(lang=context_lang, logging_level='WARN', processors='tokenize')
    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir, 'cache',
        "cached_{}_{}_{}_{}_{}_dep".format(
            evaluate,
            context_lang,
            query_lang,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        # logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        processor = SquadV1Processor()
        if evaluate == 'dev':
            args.dev_file = '../data/dev/dev-context-{}-question-{}.json'.format(context_lang, query_lang)
            # logger.info("Creating features from dataset file at %s", args.dev_file)
            examples = processor.get_dev_examples(args.data_dir, filename=args.dev_file)
        elif evaluate == 'train':
            # logger.info("Creating features from dataset file at %s", args.train_file)
            examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
        elif evaluate == 'test':
            args.test_file = '../data/test/test-context-{}-question-{}.json'.format(context_lang, query_lang)
            # logger.info("Creating features from dataset file at %s", args.test_file)
            examples = processor.get_dev_examples(args.data_dir, filename=args.test_file)
        elif evaluate == 'xquad':
            args.xquad_file = '../data/xquad/xquad.{}.json'.format(context_lang)
            examples = processor.get_dev_examples(args.data_dir, filename=args.xquad_file)

        features, dataset = squad_convert_examples_to_features_sent(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True if evaluate=='train' else False,
            stanza_nlp=stanza_nlp,
            stanza_nlp_pos=stanza_nlp_,
            sspliter=sspliter,
            language=context_lang,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert

def squad_convert_examples_to_features_sent(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    stanza_nlp,
    stanza_nlp_pos,
    sspliter,
    language,
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
    sent_level=True
):
    features = []
    unique_id = 1000000000
    example_index = 0
    for (example_index, example) in enumerate(tqdm(examples, desc="feature_converting")):
        # if example_index<719:
        #     continue
        new_features = squad_convert_example_to_features_sent(example=example,
                                                              tokenizer=tokenizer,
                                                              max_seq_length=max_seq_length,
                                                              doc_stride=doc_stride,
                                                              max_query_length=max_query_length,
                                                              is_training=is_training,
                                                              stanza_nlp=stanza_nlp,
                                                              stanza_nlp_pos=stanza_nlp_pos,
                                                              sspliter=sspliter,
                                                              language=language)
        for i in new_features:
            i.example_index = example_index
        features.extend((new_features))

    for i in features:
        i.unique_id = unique_id
        unique_id += 1

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)
    all_pos_feature = torch.tensor([f.pos_feature for f in features], dtype=torch.long)
    all_pos_feature_old = torch.tensor([f.pos_feature_old for f in features], dtype=torch.long)
    all_dep_graph_coo = torch.tensor([f.dep_graph_coo for f in features], dtype=torch.long)
    all_dep_graph_etype = torch.tensor([f.dep_graph_etype for f in features], dtype=torch.long)

    if not is_training:
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_feature_index,
            all_cls_index,
            all_p_mask,
            all_pos_feature,
            all_pos_feature_old,
            all_dep_graph_coo,
            all_dep_graph_etype
        )
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_input_ids,
            all_attention_masks,
            all_token_type_ids,
            all_start_positions,
            all_end_positions,
            all_cls_index,
            all_p_mask,
            all_is_impossible,
            all_pos_feature,
            all_pos_feature_old,
            all_dep_graph_coo,
            all_dep_graph_etype
        )
    return features, dataset

def get_clean_doc(doc):
    subword_set = []
    this_num = None
    this_word = None
    for i, word in enumerate(doc):
        if word.startswith('##'):
            this_num += 1
            this_word += word.strip('##')
        else:
            if this_num is not None:
                subword_set.append((this_word, this_num))
            this_num = 1
            this_word = word
    subword_set.append((this_word, this_num))
    assert sum([i[1] for i in subword_set])==len(doc)
    return subword_set

def get_ssplit_string(clean_doc, sspliter, language):
    clean_list = []
    window = 4
    clean_string = ' '.join(clean_doc)
    if language == 'vi':
        clean_string = re.sub(' (\W)', r'\1', clean_string)
        clean_string = re.sub('(\W) ', r'\1', clean_string)
    sspliter_clean_string = sspliter(clean_string)
    sentences = [sent.text for sent in sspliter_clean_string.sentences]
    now_list = []
    now_sent = ''
    now_sent_idx = 0
    best_ratio = 0
    for i, word in enumerate(clean_doc):
        ratio = fuzz.ratio(now_sent + word, sentences[now_sent_idx])
        if ratio >= best_ratio:
            now_list.append(word)
            best_ratio = ratio
            now_sent = now_sent + word + ' '
        else:
            try_window = []
            temp_now_sent = now_sent + word
            if i + window <= len(clean_doc):
                for j in range(1, window):
                    temp_now_sent = temp_now_sent + ' ' + clean_doc[i + j]
                    try_ratio = fuzz.ratio(temp_now_sent, sentences[now_sent_idx])
                    try_window.append(try_ratio)
            else:
                try_window = [0]
            if max(try_window) < best_ratio:
                clean_list.append(now_list)
                now_sent = word + ' '
                now_sent_idx += 1
                best_ratio = 0
                now_list = []
                now_list.append(word)
                if now_sent_idx == len(sentences):
                    now_list.extend(clean_doc[i+1:])
                    break
            else:
                now_list.append(word)
                best_ratio = ratio
                now_sent = now_sent + word + ' '


    clean_list.append(now_list)
    # for sentence in sspliter_clean_string.sentences:
    #     clean_string = clean_string.replace(sentence.text, sentence.text + '\n')
    # clean_string = re.sub('\n(\S+) ', r'\1'+'\n ', clean_string) # Sometimes \n in one word
    return clean_list

def extract_UD(span, tokenizer, stanza_nlp, sspliter, language):
    special_token = [ "[PAD]", "[CLS]", "[MASK]"]
    token_ids = span["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    original_doc = [token for token in tokens if token not in special_token]   # Remove special tokens
    original_doc = original_doc[:-1]   # Remove last [SEP], now is text1 [SEP] text2

    clean_set = get_clean_doc(original_doc)
    clean_doc = [i[0] for i in clean_set]

    # Get clean string after sentence spliting!
    sspliter_string = get_ssplit_string(clean_doc, sspliter, language)

    process_doc = stanza_nlp(sspliter_string)

    # Extract POS tagging label!
    pos_label = []
    tagging = []
    pos_label.append(0)  # [CLS]
    for sentence in process_doc.sentences:
        for word in sentence.words:
            tagging.append((word.text, word.pos))
    for i in range(len(tagging)):
        assert tagging[i][0] == clean_doc[i]
        item = Pos2idx[tagging[i][1]]
        if tagging[i][0] == '[SEP]':   # there is one [SEP] here
            pos_label.append(0)
        else:
            pos_label.extend([item]*clean_set[i][1])
    num_zeros = tokens.count("[PAD]") + 1  # [SEP] after passage and all [PAD]s
    pos_label.extend([0] * num_zeros)
    assert len(pos_label) == len(token_ids)


    # Extract dependency!
    head_list = []
    tail_list = []
    type_list = []

    token2position = []    # get token to position information
    token2position.append(('[CLS]', 0, 1))
    last_position = 1
    for i, item in enumerate(clean_set):
        start = last_position
        end = last_position + item[1]
        token2position.append((item[0], start, end))
        last_position = end
    assert last_position - 1 == len(original_doc)

    sentence_offset = 0
    for sentence in process_doc.sentences:
        for i, word in enumerate(sentence.words):
            if word.deprel != 'root' and word.text != '[SEP]':
            # Head to tail link && tail to head link
                head = sentence.words[word.head-1].text
                head_id = word.head + sentence_offset
                tail = word.text
                tail_id = i + 1 + sentence_offset
                try:
                    link_id = Rel2idx[word.deprel.split(':')[0]]
                    inv_link_id = link_id + 37
                except:
                    # print(word.deprel)  # sometimes strange deprel may appear
                    link_id = 1
                    inv_link_id = 1
                assert token2position[head_id][0] == head
                assert token2position[tail_id][0] == tail
                for j in range(token2position[head_id][1], token2position[head_id][2]):
                    for k in range(token2position[tail_id][1], token2position[tail_id][2]):
                        head_list.append(j)
                        tail_list.append(k)
                        type_list.append(link_id)

                        head_list.append(k)
                        tail_list.append(j)
                        type_list.append(inv_link_id)
            # Self-loop
                for k in range(token2position[tail_id][1], token2position[tail_id][2]):
                    head_list.append(k)
                    tail_list.append(k)
                    type_list.append(1)
            elif word.deprel == 'root':
                tail = word.text
                tail_id = i + 1 + sentence_offset
                assert token2position[tail_id][0] == tail
                for k in range(token2position[tail_id][1], token2position[tail_id][2]):
                    head_list.append(k)
                    tail_list.append(k)
                    type_list.append(36)
            elif word.text == '[SEP]':
                tail = word.text
                tail_id = i + 1 + sentence_offset
                assert token2position[tail_id][0] == tail
                for k in range(token2position[tail_id][1], token2position[tail_id][2]):
                    head_list.append(k)
                    tail_list.append(k)
                    type_list.append(1)
        sentence_offset += len(sentence.words)
    head_list.append(0)
    tail_list.append(0)
    type_list.append(1) # for [CLS]
    head_list.append(len(original_doc)+1)
    tail_list.append(len(original_doc)+1)
    type_list.append(1)  # for second [SEP]

    if language == 'ar':
        max_num = 17000
    elif language in ['zh', 'vi']:
        max_num = 5000
    elif language == 'el':
        max_num = 7100
    elif language == 'hi':
        max_num = 10000
    elif language == 'tr':
        max_num = 5000
    else:
        max_num = 4000
    if len(head_list) > max_num:
        print(len(head_list))
    while len(head_list) < max_num:
        head_list.append(0)
        tail_list.append(0)
        type_list.append(0)
    assert len(head_list) == len(tail_list) == len(type_list)

    coordinate = [head_list, tail_list]
    return pos_label, coordinate, type_list


def find_mismatch(checklist, tokens):

    for i in range(len(tokens)):
        if tokens[i] != checklist[i]:
            break

    for mis in mismatch:
        if tokens[i+mis[0]] == checklist[i+mis[1]]:
            return i, mis


def alignment(pos_label, checklist, tokens_):
    tokens = copy.deepcopy(tokens_)
    while '[SEP]' in tokens:
        tokens.remove('[SEP]')
    while '[PAD]' in tokens:
        tokens.remove('[PAD]')

    # avoid error when the mismatch is at the end of string
    pos_label.extend(['[PLACEHOLDER]']*50)
    checklist.extend(['[PLACEHOLDER]']*50)
    tokens.extend(['[PLACEHOLDER]']*50)

    while (checklist != tokens):
        # Find first mismatch part
        idx, mis = find_mismatch(checklist, tokens)
        offset = mis[1]-mis[0]
        if offset > 0:
            checklist = checklist[:idx] + \
                        tokens[idx:idx+mis[0]] + \
                        checklist[idx+mis[1]:]
            pos_label = pos_label[:idx+mis[0]] + \
                        pos_label[idx+mis[1]:]
        elif offset < 0:
            checklist = checklist[:idx] + \
                        tokens[idx:idx + mis[0]] + \
                        checklist[idx + mis[1]:]

            padding_label = pos_label[idx+mis[1]-1]
            pos_label = pos_label[:idx+mis[1]] + \
                        [padding_label]*(-offset) + \
                        pos_label[idx+mis[1]:]
        else:
            checklist = checklist[:idx] + \
                        tokens[idx:idx + mis[0]] + \
                        checklist[idx + mis[1]:]

    # Remove PLACEHOLDER tokens
    while '[PLACEHOLDER]' in pos_label:
        pos_label.remove('[PLACEHOLDER]')
    while '[PLACEHOLDER]' in checklist:
        checklist.remove('[PLACEHOLDER]')

    return pos_label, checklist

def specific_processing(original_doc, language):
    if language == 'vi':
        # original_doc = re.sub('(\d+) . (\d+) (?!%)', r'\1.\2', original_doc)
        # original_doc = re.sub(' - ', r'-', original_doc)
        # original_doc = original_doc.replace('. W .', 'UNK')
        # original_doc = original_doc.replace('. H .', 'UNK')

        original_doc = re.sub(' (\W)', r'\1', original_doc)
        original_doc = re.sub('(\W) ', r'\1', original_doc)
    # if language == 'ar':
    #     original_doc = original_doc.replace(' ','')
    return original_doc

def extract_POS(span, tokenizer, stanza_nlp, language):
    special_token = ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    token_ids = span["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    original_doc = [token for token in tokens if token not in special_token]
    pos_label = []

    # For en?
    tagging = []
    checklist = []
    original_doc = tokenizer.convert_tokens_to_string(original_doc).replace('[UNK]','UNK')
    original_doc = specific_processing(original_doc,language)
    doc = stanza_nlp(original_doc)
    for sentence in doc.sentences:
        for word in sentence.words:
            tagging.append((word.text, word.pos))
    pos_label.append(0) # [CLS]
    checklist.append("[CLS]")
    for text, pos in tagging:
        if text == 'UNK':
            text = '[UNK]'
        item = Pos2idx[pos]
        text_tokens = tokenizer.tokenize(text)
        items = len(text_tokens)*[item]
        pos_label.extend(items)
        checklist.extend(text_tokens)
    pos_label, checklist = alignment(pos_label, checklist, tokens)
    num_zeros = tokens.count("[PAD]")+1  # [SEP] after passage and all [PAD]s
    pos_label.extend([0]*num_zeros)
    checklist.append("[SEP]")
    checklist.extend(["[PAD]"]*(num_zeros-1))
    pos_label.insert(tokens.index("[SEP]"), 0) # [SEP] between query and passage
    checklist.insert(tokens.index("[SEP]"), "[SEP]")
    assert len(pos_label) == len(tokens)
    # try:
    #     assert len(pos_label) == len(tokens)
    # except:
    #     print(list(zip(tokens, checklist)))
    return pos_label




def squad_convert_example_to_features_sent(example, tokenizer, max_seq_length, doc_stride, max_query_length, is_training, stanza_nlp, stanza_nlp_pos, sspliter, language):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    spans = []

    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query if tokenizer.padding_side == "right" else span_doc_tokens,
            span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
            padding="max_length",
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        try:
            pos_feature, dep_graph_coo, dep_graph_etype = extract_UD(span, tokenizer, stanza_nlp, sspliter, language)
            pos_feature_old = extract_POS(span, tokenizer, stanza_nlp_pos, language)
        except:
            print('Skip this sample.')
            continue

        features.append(
            SquadFeatures_sent(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                pos_feature_old=pos_feature_old,
                pos_feature=pos_feature,
                dep_graph_coo=dep_graph_coo,
                dep_graph_etype=dep_graph_etype,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features

class SquadFeatures_sent(object):

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        pos_feature_old,
        pos_feature,
        dep_graph_coo,
        dep_graph_etype,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.pos_feature_old = pos_feature_old
        self.pos_feature = pos_feature
        self.dep_graph_coo = dep_graph_coo
        self.dep_graph_etype = dep_graph_etype
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id
