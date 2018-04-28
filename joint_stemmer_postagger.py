# coding=utf-8
import random
import re
import math
import sys
from collections import defaultdict
from collections import namedtuple
from pprint import pprint

import dynet as dy
import numpy as np
import time
import pickle
from datetime import datetime
import logging.config

logger = logging.getLogger(__file__)


class TurkishStemmerPOSTagger(object):
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    CONSONANT_STR = u"[bcdfgğhjklmnprsştvyzxwqBCDFGĞHJKLMNPRSŞTVYZXWQ]"
    VOWEL_STR = u"[aeıioöuüAEIİOÖUÜ]"
    WIDE_VOWELS_STR = u"[aeoöAEOÖ]"
    NARROW_VOWELS_STR = u"[uüıiUÜIİ]"
    UPPERS_STR = u"[ABCDEFGĞHIİJKLMNOÖPRSŞTUÜVYZXWQ]"

    TWO_CONSANANT_REG = re.compile(r"^.*{}{}$".format(CONSONANT_STR, CONSONANT_STR), re.UNICODE)
    START_NARROW_REGEX = re.compile(r"^{}.*$".format(NARROW_VOWELS_STR), re.UNICODE)
    REPLACE_LAST_CONSANANT_REG = re.compile(r"^(.*)({})$".format(CONSONANT_STR), re.UNICODE)
    ENDS_WITH_WIDE_WOVEL = re.compile(r"^.*{}$".format(WIDE_VOWELS_STR), re.UNICODE)
    START_WITH_UPPER = re.compile(r"^{}.*$".format(UPPERS_STR), re.UNICODE)
    GET_STEM_REGEX = re.compile(r"^(.*)\+[^\+]*\[(.*)\]$", re.UNICODE)

    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "suffixes", "postags"])
    analysis_regex = re.compile(r"^([^\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\+\^]", re.UNICODE)
    split_root_tags_regex = re.compile(r"^([^\+]+)\+(.+)$", re.IGNORECASE)
    ROOT_TRANSFORMATION_MAP = {"tıp": "tıb", "prof.": "profesör", "dr.": "doktor",
                               "yi": "ye", "ed": "et", "di": "de"}
    @classmethod
    def _create_vocab_chars(cls, sentences):
        char2id = defaultdict(int)
        char2id["<"] = len(char2id) + 1
        char2id["/"] = len(char2id) + 1
        char2id[">"] = len(char2id) + 1
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id) + 1
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id:
                            char2id[ch] = len(char2id) + 1
        return char2id

    @classmethod
    def _encode(cls, tokens, vocab):
        return [vocab[token] for token in tokens]

    @classmethod
    def _embed(cls, token, char_embedding_table):
        return [char_embedding_table[ch] for ch in token]

    @classmethod
    def _print_namedtuple(cls, nt):
        pprint(dict(nt._asdict()))

    def __init__(self, train_from_scratch=True, resume_training=False, char_representation_len=100,
                 word_lstm_rep_len=200, train_data_path="data/data.train.txt",
                 dev_data_path="data/data.dev.txt", test_data_paths=["data/data.test.txt"],
                 model_file_name=None, char2id=None, postag2id=None, case_sensitive=False,
                 postagging=True, stemming=True, unknown_tag="***UNKNOWN"):
        assert word_lstm_rep_len % 2 == 0
        self.postagging = postagging
        self.stemming = stemming
        self.case_sensitive = case_sensitive
        if train_from_scratch:
            assert train_data_path
            assert len(test_data_paths) > 0
            logger.info("Loading data...")
            self.train = self.load_data(train_data_path)
            if dev_data_path:
                self.dev = self.load_data(dev_data_path)
            else:
                self.dev = None
            self.test_paths = test_data_paths
            self.tests = []
            for test_path in self.test_paths:
                self.tests.append(self.load_data(test_path))
            logger.info("Creating or Loading Vocabulary...")
            if char2id:
                self.char2id = char2id
            else:
                self.char2id = self._create_vocab_chars(self.train)
            if postag2id:
                self.postag2id = tag2id
            else:
                self.postag2id = {unknown_tag: 0, "Noun": 1, "Verb": 2, "Adj": 3, "Adv": 4, "Pron": 5, "Conj": 6,
                                  "Interj": 7, "Punc": 8, "Num": 9, "Det": 10, "Postp": 11, "Adverb": 12, "Ques": 13}

            if not self.dev:
                train_size = int(math.floor(0.99 * len(self.train)))
                self.dev = self.train[train_size:]
                self.train = self.train[:train_size]
            self.model = dy.Model()
            self.trainer = dy.AdamTrainer(self.model)
            self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id) + 2, char_representation_len))
            self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters(
                (len(self.char2id) + 2, char_representation_len))
            self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            if stemming:
                self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
                self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
                self.fwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
                self.bwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            if postagging:
                self.pW = self.model.add_parameters((word_lstm_rep_len, len(self.postag2id)))
                self.pb = self.model.add_parameters(len(self.postag2id))
            self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
            self.train_model(model_name=model_file_name)
        else:
            logger.info("Loading Pre-Trained Model")
            if char2id:
                self.char2id = char2id
            if postag2id:
                self.postag2id = postag2id
            self.load_model(char_representation_len, word_lstm_rep_len)
            if resume_training:
                assert train_data_path
                assert len(test_data_paths) > 0
                logger.info("Loading data...")
                self.train = self.load_data(train_data_path)
                if dev_data_path:
                    self.dev = self.load_data(dev_data_path)
                else:
                    self.dev = None
                self.test_paths = test_data_paths
                self.tests = []
                for test_path in self.test_paths:
                    self.tests.append(self.load_data(test_path))
                self.train_model(model_name="model")

    def _get_tags_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return self.tag_seperator_regex.split(analysis[2:])
        else:
            return self.tag_seperator_regex.split(self.analysis_regex.sub(r"\2", analysis))

    def _get_root_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return "+"
        else:
            return self.analysis_regex.sub(r"\1", analysis)

    def _get_pos_from_analysis(self, analysis):
        tags = self._get_tagsstr_from_analysis(analysis)
        if "^" in tags:
            tags = tags[tags.rfind("^") + 4:]
        return tags.split("+")[0]

    def _get_tagsstr_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return analysis[2:]
        else:
            return self.split_root_tags_regex.sub(r"\2", analysis)

    @staticmethod
    def add_candidate(candidate_stem, candidate_suffix, candidate_stem_list,
                      candidate_suffix_list, case_sensitive=False):
        if TurkishStemmerPOSTagger.tr_lower(candidate_stem) in \
                TurkishStemmerPOSTagger.ROOT_TRANSFORMATION_MAP:
            candidate_stem = TurkishStemmerPOSTagger.ROOT_TRANSFORMATION_MAP[
                TurkishStemmerPOSTagger.tr_lower(candidate_stem)
            ]
        if not case_sensitive:
            candidate_stem = TurkishStemmerPOSTagger.tr_lower(candidate_stem)
            candidate_suffix = TurkishStemmerPOSTagger.tr_lower(candidate_suffix)
        candidate_stem_list.append(candidate_stem)
        candidate_suffix_list.append(candidate_suffix)

    @staticmethod
    def generate_candidate_roots_and_suffixes(surface_word, turkish_transformations=True, case_sensitive=False):
        candidate_roots = []
        candidate_suffixes = []
        for i in range(1, len(surface_word)):
            candidate_root = surface_word[:i]
            candidate_suffix = surface_word[i:]
            TurkishStemmerPOSTagger.add_candidate(candidate_root,
                                                  candidate_suffix,
                                                  candidate_roots,
                                                  candidate_suffixes,
                                                  case_sensitive=case_sensitive)
            if candidate_root == "ban" or candidate_root == "Ban":
                TurkishStemmerPOSTagger.add_candidate("ben", candidate_suffix,
                                                      candidate_roots,
                                                      candidate_suffixes,
                                                      case_sensitive=case_sensitive)
            if candidate_root == "san" or candidate_root == "San":
                TurkishStemmerPOSTagger.add_candidate("sen", candidate_suffix,
                                                      candidate_roots,
                                                      candidate_suffixes,
                                                      case_sensitive=case_sensitive)

            if turkish_transformations:
                candidate_root = TurkishStemmerPOSTagger.tr_lower(candidate_root)
                if len(candidate_root) > 2 and len(candidate_suffix) > 0 and candidate_root[-1] == candidate_root[-2]:
                    TurkishStemmerPOSTagger.add_candidate(candidate_root[:-1],
                                                          candidate_suffix,
                                                          candidate_roots,
                                                          candidate_suffixes,
                                                          case_sensitive=case_sensitive)
                if len(candidate_root) > 2 and \
                        TurkishStemmerPOSTagger.TWO_CONSANANT_REG.match(candidate_root) and \
                        TurkishStemmerPOSTagger.START_NARROW_REGEX.match(candidate_suffix):
                    candidate_root2 = TurkishStemmerPOSTagger.REPLACE_LAST_CONSANANT_REG \
                        .sub(r"\1{}\2".format(candidate_suffix[0]), candidate_root)
                    TurkishStemmerPOSTagger.add_candidate(candidate_root2,
                                                          candidate_suffix,
                                                          candidate_roots,
                                                          candidate_suffixes,
                                                          case_sensitive=case_sensitive)
                if len(candidate_root) > 2 \
                        and re.match(r".*[ıiuü]$", candidate_root) and u"yor" in candidate_suffix:
                    candidate_root2 = re.sub(r"^(.*)([uı])$", r"\1a", candidate_root)
                    candidate_root2 = re.sub(r"^(.*)([üi])$", r"\1e", candidate_root2)
                    TurkishStemmerPOSTagger.add_candidate(candidate_root2,
                                                          candidate_suffix,
                                                          candidate_roots,
                                                          candidate_suffixes,
                                                          case_sensitive=case_sensitive)
                if len(candidate_root) > 1 and re.match(r"^.*[bcdğBCDĞ]$", candidate_root):
                    candidate_root2 = re.sub(r"^(.*)b$", r"\1p", candidate_root)
                    candidate_root2 = re.sub(r"^(.*)B$", r"\1P", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)c$", r"\1ç", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)C$", r"\1Ç", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)d$", r"\1t", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)D$", r"\1T", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)ğ$", r"\1k", candidate_root2)
                    candidate_root2 = re.sub(r"^(.*)Ğ$", r"\1K", candidate_root2)
                    TurkishStemmerPOSTagger.add_candidate(candidate_root2,
                                                          candidate_suffix,
                                                          candidate_roots,
                                                          candidate_suffixes,
                                                          case_sensitive=case_sensitive)
        TurkishStemmerPOSTagger.add_candidate(surface_word,
                                              "",
                                              candidate_roots,
                                              candidate_suffixes,
                                              case_sensitive=case_sensitive)
        candidate_roots.append(surface_word)
        candidate_suffixes.append("")
        if TurkishStemmerPOSTagger.START_WITH_UPPER.match(surface_word) and case_sensitive:
            candidate_roots.append(TurkishStemmerPOSTagger.tr_lower(surface_word))
            candidate_suffixes.append("")
        return candidate_roots, candidate_suffixes

    def load_data(self, file_path, max_sentence=1000):
        sentence = []
        sentences = []
        with open(file_path, 'r') as f:
            for line in f:
                trimmed_line = line.strip(" \r\n\t")
                if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                    sentence = []
                elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        if len(sentences) > max_sentence:
                            return sentences
                elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith("</DOC>") or \
                        trimmed_line.startswith("<TITLE>") or trimmed_line.startswith("</TITLE>"):
                    pass
                else:
                    parses = re.split(r"[\t ]", trimmed_line)
                    surface = parses[0]
                    candidate_roots, candidate_suffixes = \
                        TurkishStemmerPOSTagger.generate_candidate_roots_and_suffixes(surface,
                                                                                      case_sensitive=self.case_sensitive)
                    assert len(candidate_roots) == len(candidate_suffixes)
                    analyzes = parses[1:]
                    gold_root = self._get_root_from_analysis(analyzes[0]).lower()
                    roots = []
                    suffixes = []
                    roots.append(gold_root)
                    gold_suffix = surface[len(gold_root):]
                    suffixes.append(gold_suffix)
                    for candidate_root, candidate_suffix in zip(candidate_roots, candidate_suffixes):
                        if candidate_root != gold_root and candidate_suffix != gold_suffix:
                            roots.append(candidate_root)
                            suffixes.append(candidate_suffix)
                    postags = []
                    for analysis in analyzes:
                        cur_postag = self._get_pos_from_analysis(analysis)
                        if cur_postag not in postags:
                            postags.append(cur_postag)
                    current_word = self.WordStruct(surface, roots, suffixes, postags)
                    sentence.append(current_word)
        return sentences

    def propogate(self, sentence):
        dy.renew_cg()
        fwdRNN_surface_init = self.fwdRNN_surface.initial_state()
        bwdRNN_surface_init = self.bwdRNN_surface.initial_state()
        if self.stemming:
            fwdRNN_root_init = self.fwdRNN_root.initial_state()
            bwdRNN_root_init = self.bwdRNN_root.initial_state()
            fwdRNN_suffix_init = self.fwdRNN_suffix.initial_state()
            bwdRNN_suffix_init = self.bwdRNN_suffix.initial_state()
        fwdRNN_context_init = self.fwdRNN_context.initial_state()
        bwdRNN_context_init = self.bwdRNN_context.initial_state()
        if self.postagging:
            W = dy.parameter(self.pW)
            b = dy.parameter(self.pb)

        # CONTEXT REPRESENTATIONS
        surface_words_rep = []
        for index, word in enumerate(sentence):
            encoded_surface_word = self._encode(word.surface_word, self.char2id)
            surface_word_char_embeddings = self._embed(encoded_surface_word, self.SURFACE_CHARS_LOOKUP)
            fw_exps_surface_word = fwdRNN_surface_init.transduce(surface_word_char_embeddings)
            bw_exps_surface_word = bwdRNN_surface_init.transduce(reversed(surface_word_char_embeddings))
            surface_word_rep = dy.concatenate([fw_exps_surface_word[-1], bw_exps_surface_word[-1]])
            surface_words_rep.append(surface_word_rep)
        fw_exps_context = fwdRNN_context_init.transduce(surface_words_rep)
        bw_exps_context = bwdRNN_context_init.transduce(reversed(surface_words_rep))
        root_scores = []
        postag_scores = []
        # Stem and POS REPRESENTATIONS
        for index, word in enumerate(sentence):
            if self.stemming:
                encoded_roots = [self._encode(root, self.char2id) for root in word.roots]
                encoded_suffixes = [self._encode(suffix, self.char2id) for suffix in word.suffixes]
                roots_embeddings = [self._embed(root, self.ROOT_CHARS_LOOKUP) for root in encoded_roots]
                suffix_embeddings = [self._embed(suffix, self.ROOT_CHARS_LOOKUP) for suffix in encoded_suffixes]
                root_stem_representations = []
                for root_embedding, suffix_embedding in zip(roots_embeddings, suffix_embeddings):
                    fw_exps_root = fwdRNN_root_init.transduce(root_embedding)
                    bw_exps_root = bwdRNN_root_init.transduce(reversed(root_embedding))
                    root_representation = dy.rectify(dy.concatenate([fw_exps_root[-1], bw_exps_root[-1]]))
                    if len(suffix_embedding) != 0:
                        fw_exps_suffix = fwdRNN_suffix_init.transduce(suffix_embedding)
                        bw_exps_suffix = bwdRNN_suffix_init.transduce(reversed(suffix_embedding))
                        suffix_representation = dy.rectify(dy.concatenate([fw_exps_suffix[-1], bw_exps_suffix[-1]]))
                        root_stem_representations.append(dy.rectify(dy.esum([root_representation, suffix_representation])))
                    else:
                        root_stem_representations.append(root_representation)

            left_context_rep = fw_exps_context[index]
            right_context_rep = bw_exps_context[len(sentence) - index - 1]
            context_rep = dy.tanh(dy.esum([left_context_rep, right_context_rep]))
            if self.stemming and self.postagging:
                root_scores.append(
                    (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * dy.concatenate(root_stem_representations, 1))[0])
                postag_scores.append(
                    (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * W + b)[0]
                )
            elif self.stemming:
                root_scores.append(
                    (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * dy.concatenate(root_stem_representations,
                                                                                            1))[0])
            elif self.postagging:
                postag_scores.append(
                    (dy.reshape(context_rep, (1, context_rep.dim()[0][0])) * W + b)[0]
                )

        return root_scores, postag_scores

    def get_loss(self, sentence):
        root_scores, postag_scores = self.propogate(sentence)
        errs = []
        if self.postagging and self.stemming:
            for i, (root_score, postag_score) in enumerate(zip(root_scores, postag_scores)):
                root_err = dy.pickneglogsoftmax(root_score, 0)
                errs.append(root_err)
                gold_postag = sentence[i].postags[0]
                if gold_postag in self.postag2id:
                    pos_err = dy.pickneglogsoftmax(postag_score, self.postag2id[sentence[i].postags[0]])
                    errs.append(pos_err)
        elif self.stemming:
            for i, root_score in enumerate(root_scores):
                root_err = dy.pickneglogsoftmax(root_score, 0)
                errs.append(root_err)
        elif self.postagging:
            for i, postag_score in enumerate(postag_scores):
                gold_postag = sentence[i].postags[0]
                if gold_postag in self.postag2id:
                    pos_err = dy.pickneglogsoftmax(postag_score, self.postag2id[sentence[i].postags[0]])
                    errs.append(pos_err)
        return dy.esum(errs)

    def predict_indices(self, sentence):
        selected_root_indices = []
        selected_postag_indices = []
        root_scores, postag_scores = self.propogate(sentence)
        if self.postagging and self.stemming:
            for i, (root_score, postag_score) in enumerate(zip(root_scores, postag_scores)):
                root_probs = dy.softmax(root_score).npvalue()
                selected_root_index = np.argmax(root_probs)
                selected_root_indices.append(selected_root_index)
                postag_probs = dy.softmax(postag_score)
                selected_postag_indices.append(np.argmax(postag_probs.npvalue()))
        elif self.stemming:
            for i, root_score in enumerate(root_scores):
                root_probs = dy.softmax(root_score).npvalue()
                selected_root_index = np.argmax(root_probs)
                selected_root_indices.append(selected_root_index)
        elif self.postagging:
            for i, postag_score in enumerate(postag_scores):
                postag_probs = dy.softmax(postag_score)
                selected_postag_indices.append(np.argmax(postag_probs.npvalue()))
        return selected_root_indices, selected_postag_indices

    def calculate_acc(self, sentences, labels=None):
        reverse_postag_index = {v: k for k, v in self.postag2id.items()}
        root_corrects = 0
        postag_corrects = 0
        both_corrects = 0
        total = 0
        if not labels:
            labels = [[0 for w in sentence] for sentence in sentences]
        for sentence, sentence_labels in zip(sentences, labels):
            selected_root_indices, selected_postag_indices = self.predict_indices(sentence)
            if self.stemming:
                root_corrects += [1 for l1, l2 in zip(sentence_labels, selected_root_indices) if l1 == l2].count(1)
            if self.postagging:
                postag_corrects += [1 for w, p_index in zip(sentence, selected_postag_indices) if
                                w.postags[0] == reverse_postag_index[p_index]].count(1)

            if self.stemming and self.postagging:
                both_corrects += [x1 + x2 for x1, x2 in zip(
                    [1 for l1, l2 in zip(sentence_labels, selected_root_indices) if l1 == l2],
                    [1 for w, p_index in zip(sentence, selected_postag_indices) if
                     w.postags[0] == reverse_postag_index[p_index]]
                )].count(2)
            elif self.postagging:
                both_corrects = postag_corrects
            elif self.stemming:
                both_corrects = root_corrects
            total += len(sentence)
        return (root_corrects * 1.0) / total, (postag_corrects * 1.0) / total, (both_corrects * 1.0) / total

    def train_model(self, model_name="model", early_stop=True, num_epoch=20):
        max_acc = 0.0
        epoch_loss = 0
        for epoch in range(num_epoch):
            random.shuffle(self.train)
            t1 = datetime.now()
            count = 0
            for i, sentence in enumerate(self.train, 1):
                loss_exp = self.get_loss(sentence)
                cur_loss = loss_exp.scalar_value()
                epoch_loss += cur_loss
                loss_exp.backward()
                self.trainer.update()
                if i > 0 and i % 100 == 0:  # print status
                    t2 = datetime.now()
                    delta = t2 - t1
                    logger.info("loss = {}  /  {} instances finished in  {} seconds".format(epoch_loss / (i * 1.0), i, delta.seconds))
                count = i
            t2 = datetime.now()
            delta = t2 - t1
            logger.info("epoch {} finished in {} minutes. loss = {}".format(epoch, delta.seconds / 60.0, epoch_loss / count * 1.0))
            epoch_loss = 0
            root_acc, postag_acc, both_acc = self.calculate_acc(self.dev)
            logger.info("Calculating Accuracy on dev set")
            logger.info("Root accuracy on dev set:{}\nPostag accuracy on dev set:{} Joint Accuracy on dev set:{}"\
                .format(root_acc, postag_acc, both_acc))
            if both_acc > max_acc:
                max_acc = both_acc
                logger.info("Max accuracy increased, saving model...")
                self.save_model(model_name)
            elif early_stop and max_acc > both_acc:
                logger.info("Max accuracy did not incrase, early stopping!")
                with open("results.txt", "a") as f:
                    f.write(write_to_file)
                    f.write("\n")
                    f.flush()
                break

            logger.info("Calculating Accuracy on test sets")
            write_to_file = "\nPerformance of {} model\n".format(model_name)
            for q in range(len(self.test_paths)):
                logger.info("Calculating Accuracy on test set: {}".format(self.test_paths[q]))
                root_acc, postag_acc, both_acc = self.calculate_acc(self.tests[q])
                logger.info("Root accuracy on test set:{}\nPostag accuracy on test set:{} Joint Accuracy on test set:{}"\
                    .format(root_acc, postag_acc, both_acc))
                write_to_file += "Calculating Accuracy on test set: {}\n".format(self.test_paths[q])
                write_to_file += "Root accuracy on test set:{}\nPostag accuracy on test set:{} " \
                                 "Joint Accuracy on test set:{}\n\n".format(root_acc, postag_acc, both_acc)

    def save_model(self, model_name):
        self.model.save("resources/nlp/turkish/stemmer/"+model_name+".model")
        with open("resources/nlp/turkish/stemmer/"+model_name+".char2id", "w") as f:
            pickle.dump(self.char2id, f)
        with open("resources/nlp/turkish/stemmer/"+model_name+".tag2id", "w") as f:
            pickle.dump(self.postag2id, f)

    def load_model(self, char_representation_len, word_lstm_rep_len, postagging=True, stemming=True):
        with open("resources/nlp/turkish/stemmer/model.char2id", "rb") as f:
            self.char2id = pickle.load(f)
        with open("resources/nlp/turkish/stemmer/model.tag2id", "rb") as f:
            self.postag2id = pickle.load(f)

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        self.SURFACE_CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.char2id) + 2, char_representation_len))
        self.ROOT_CHARS_LOOKUP = self.model.add_lookup_parameters(
            (len(self.char2id) + 2, char_representation_len))
        self.fwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        self.bwdRNN_surface = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        if stemming:
            self.fwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_root = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.fwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
            self.bwdRNN_suffix = dy.LSTMBuilder(1, char_representation_len, word_lstm_rep_len / 2, self.model)
        if postagging:
            self.pW = self.model.add_parameters((word_lstm_rep_len, len(self.postag2id)))
            self.pb = self.model.add_parameters(len(self.postag2id))
        self.fwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.bwdRNN_context = dy.LSTMBuilder(1, word_lstm_rep_len, word_lstm_rep_len, self.model)
        self.model.populate("resources/nlp/turkish/stemmer/model")

    @classmethod
    def create_from_existed_model(cls, char2id=None, tag2id=None):
        return TurkishStemmerPOSTagger(train_from_scratch=False,
                                       char2id=char2id, postag2id=tag2id)

    def predict(self, tokens):
        reverse_postag_index = {v: k for k, v in self.postag2id.items()}
        sentence = []
        for token in tokens:
            candidate_roots, candidate_suffixes = \
                TurkishStemmerPOSTagger.generate_candidate_roots_and_suffixes(token,
                                                                              case_sensitive=self.case_sensitive)
            assert len(candidate_roots) == len(candidate_suffixes)
            current_word = self.WordStruct(token, candidate_roots, candidate_suffixes, None)
            sentence.append(current_word)

        selected_root_indices, selected_postag_indices = self.predict_indices(sentence)
        res = []
        for w, root_i, postag_i in zip(sentence, selected_root_indices, selected_postag_indices):
            res.append(w.roots[root_i] + "+" + w.suffixes[root_i]
                       + "[" + reverse_postag_index[postag_i] + "]")
        return res

    def get_stems(self, tokens):
        stem_postags = self.predict(tokens)
        stems = []
        for stem_postag in stem_postags:
            stem = TurkishStemmerPOSTagger.GET_STEM_REGEX.sub(r"\1", stem_postag).strip()
            if len(stem) > 0:
                stems.append(stem)
        return stems

    def get_stem_and_postags(self, tokens):
        stem_postags = self.predict(tokens)
        stems = []
        postags = []
        for stem_postag in stem_postags:
            stem = TurkishStemmerPOSTagger.GET_STEM_REGEX.sub(r"\1", stem_postag).strip()
            postag = TurkishStemmerPOSTagger.GET_STEM_REGEX.sub(r"\2", stem_postag).strip()
            if len(stem) > 0:
                stems.append(stem)
                postags.append(postag)
        return stems, postags

    def predict_file(self, file_path, output_file_path):
        with open(output_file_path, "w") as w:
            with open(file_path, "r") as f:
                for line in f:
                    trimmed_line = line.strip(" \r\n\t")
                    if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                        w.write("<S>\n")
                        sentence = []
                        gold_sentence = []
                    elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                        if len(sentence) > 0:
                            res = self.predict(sentence)
                            w.write("\n".join(["\t"+gold_sentence[i]+"\t"+res[i]+
                                               "\t" + str(re.sub(r"^([^\+]+)\+.*$", r"\1", gold_sentence[i]) ==
                                                          re.sub(r"^([^\+]+)\+.*$", r"\1", res[i])) for i in
                                               range(len(gold_sentence))]))
                            w.write("\n</S>\n")
                            w.flush()
                    elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith(
                            "</DOC>") or trimmed_line.startswith(
                            "<TITLE>") or trimmed_line.startswith("</TITLE>"):
                        w.write("\n".format(trimmed_line[:trimmed_line.find(" ")]))
                    else:
                        parses = re.split(r"[\t ]", trimmed_line)
                        surface = parses[0]
                        gold_sentence.append(parses[1])
                        sentence.append(surface)

    @staticmethod
    def tr_lower(self):
        self = re.sub(r"İ", "i", self)
        self = re.sub(r"I", "ı", self)
        self = re.sub(r"Ç", "ç", self)
        self = re.sub(r"Ş", "ş", self)
        self = re.sub(r"Ü", "ü", self)
        self = re.sub(r"Ğ", "ğ", self)
        self = self.lower()  # for the rest use default lower
        return self


if __name__ == "__main__":
    char2id = None
    tag2id = None
    stemmer = TurkishStemmerPOSTagger.create_from_existed_model()
    while 1:
        try:
            sys.stdout.write("\nInput a Turkish sentence:\n")
            sentence = sys.stdin.readline().strip()
            stems, postags = stemmer.get_stem_and_postags(sentence.split(" "))
            logger.info(" ".join([stem + "+" + postag for stem, postag in zip(stems, postags)]))
        except KeyboardInterrupt:
            break