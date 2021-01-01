import os.path
import json
import bz2
import pickle
import random
import re
import copy
import math
from urllib.parse import unquote
import numpy as np
import torch
from torch.utils.data import Dataset
from flair.embeddings import RoBERTaEmbeddings
from flair.data import Sentence
# from s2v_embedding_models.s2v_gammaTransformer.s2v_gammaTransformer import generate_s2v
# import spacy
from tqdm import tqdm
from datetime import datetime

random.seed(0)
batch_train_data = []
batch_train_data_size = 0
batch_valid_data = []
batch_valid_data_size = 0
batch_full_data = {}
batch_full_mask_data = {}
batch_file_list = []
init_cnt = 0

train_title_list = []
valid_title_list  = []
train_title_weight = []
valid_title_weight = []

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def find_html_links_from_wikipage(text):
    links = []
    # search_results = re.findall('\shref.*?>', text)
    search_results = re.findall("href=\"view-source:https://en.wikipedia.org/wiki/.*?>", text)
    search_results = re.findall("<a href=\"https://en.wikipedia.org/wiki/.*?\s", text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"https://en.wikipedia.org/wiki/", "")[:-2]))
    return links

class WikiS2vCorrectionBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.valid = valid

        self.datasets_dir = "/home/kalickid/Projects/github/s2v_linker/datasets/hotpotqa/"
        self.batch_dir = '../s2v_mem_correction/train/datasets/'

        # articles = self._get_input_articles_list_from_vital_articles()
        # self._create_batch_words_emb(articles)
        # self._create_batch_s2v()
        # exit(1)

        self.batch_files = []
        self.batch_idx = 0
        self.true_s2v_rate = self.config['training']['memory_true_s2v_initial_rate']
        self._init_batch()
    
    def _get_input_articles_list_from_vital_articles(self):
        self.vital_articles_dump_dir = '../doc_Transformer/datasets/wiki/'
        articles = set()
        for file in sorted(os.listdir(self.vital_articles_dump_dir)):
            if file.split(".")[-1] == 'html':
                cnt = 0
                if 'html' in file:
                    with open(self.vital_articles_dump_dir+file, "r") as f:
                        for link in find_html_links_from_wikipage(f.read()):
                            key_words_list = [":", "Main_Page"]
                            if all(word not in link for word in key_words_list):
                                article = link.replace("_", " ")
                                articles.update([article.lower()])
                                cnt += 1
        print('Number of input articles: '+ str(len(articles)))
        return articles

    def _embed_sentences(self, text):
        article = []
        masked_article = []
        for p_idx, paragraph in enumerate(text):
            for l_idx, line in enumerate(paragraph):
                sentence = line
                sentence = remove_html_tags(sentence)
                if len(sentence) > 0:
                    sentence_emb, words = self._process_sentences([sentence])
                    article.append({
                        'sentence': sentence,
                        'sentence_emb': sentence_emb,
                        'sentence_words': words,
                        'paragraph_idx': p_idx,
                        'line_idx': l_idx
                    })
                    # mask_sent_split = []
                    # for word in sentence.strip().split(" "):
                    #     if random.randint(0, 99) < 10:
                    #         mask_sent_split.append("<mask>")
                    #     else:
                    #         mask_sent_split.append(word)
                    # masked_sentence = " ".join(mask_sent_split)
                    # sentence_emb, words = self._process_sentences([masked_sentence])
                    # masked_article.append({
                    #     'sentence': sentence,
                    #     'sentence_emb': sentence_emb,
                    #     'sentence_words': words,
                    #     'paragraph_idx': p_idx,
                    #     'line_idx': l_idx
                    # })

        return article, masked_article
    
    def _create_batch_words_emb(self, articles):
        self.embedding = RoBERTaEmbeddings(
            pretrained_model_name_or_path="roberta-large",
            layers="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20," +
                   "21,22,23,24",
            pooling_operation="mean", use_scalar_mix=True)
        all_articles_lc = [x.lower() for x in articles]
        all_sentences = 0
        for folder in sorted(os.listdir(self.datasets_dir)):
            for file in sorted(os.listdir(self.datasets_dir+folder)):
                if file.split(".")[-1] == 'bz2':
                    if not os.path.isfile(self.batch_dir + folder + file.replace("wiki", "").replace(".bz2", "") + "_" + 'articles_wEmb.pickle'):
                        print(self.datasets_dir+folder+'/'+file)
                        article_dict = {}
                        masked_article_dict = {}
                        with bz2.BZ2File(self.datasets_dir+folder+"/"+file, "r") as fp:
                            for line in fp:
                                data = json.loads(line)
                                title = data['title'].lower()
                                if (title in all_articles_lc) and (title != "oath of office") and (title != "punjabi language") and \
                                   (title != "surname") and (title != "florence la badie") and (title != "plasmodium") and (title != "anambra state") and \
                                   (title != "norodom sihamoni") and (title != "hefei"):
                                    text = data['text'][1:-1] # skip first and last paragraph
                                    sent_cnt = 0
                                    for paragraph in text:
                                        sent_cnt += len(paragraph)
                                    if sent_cnt > 8:
                                        print("\t", title, sent_cnt)
                                        try:
                                            article, masked_article = self._embed_sentences(text)
                                            article_dict[title] = article
                                            masked_article_dict[title] = masked_article
                                        except IndexError:
                                            print('\tIndexError')
                        if len(article_dict) > 0:
                            pickle.dump(article_dict, open(self.batch_dir + folder + file.replace("wiki", "").replace(".bz2", "") + "_" + 'articles_wEmb.pickle', 'wb'))
                            # pickle.dump(masked_article_dict, open(self.batch_dir + folder + file.replace("wiki", "").replace(".bz2", "") + "_" + 'masked_articles_wEmb.pickle', 'wb'))

    def _process_sentences(self, sentences):
        sentences_emb = []
        words = []
        for sentence in sentences:
            sentence = " ".join(sentence.split())
            sent = sentence.strip()
            if len(sent.strip()) == 0:
                sent = 'empty'
            if len(sent.split(" ")) > 220:
                print(len(sent.split(" ")))
                sent = sent[:512]
            try:
                sent = Sentence(sent)
                self.embedding.embed(sent)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                words.append([t.text for t in sent])
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
            except IndexError:
                print('IndexError')
                print(sentence)
                sentence_emb = [np.array(t.embedding).astype(np.float16)
                                for t in sent]
                sentences_emb.append(np.array(sentence_emb).astype(np.float16))
        sentences_emb_short = sentences_emb
        return sentences_emb_short, words

    def _create_batch_s2v(self):
        for file in sorted(os.listdir(self.batch_dir)):
            if (file.split(".")[-1] == 'pickle') and (not os.path.isfile(self.batch_dir + file.replace('_wEmb.', '_s2v.'))) and \
               ('_wEmb' in file):
                print(file)
                data = pickle.load(open(self.batch_dir+file, 'rb'))
                print(len(data))
                for title in data:
                    print(title)
                    pbar = tqdm(total=len(data[title]), dynamic_ncols=True)
                    for sentence in data[title]:
                        s2v_gammaTransformer = self._generate_s2v(sentence['sentence_emb'])
                        del sentence['sentence_emb']
                        sentence['sentence_vect_gTr'] = s2v_gammaTransformer[0]
                        pbar.update(1)
                    pbar.close()
                if len(data) > 0:
                    pickle.dump(data, open(self.batch_dir + file.replace('_wEmb.', '_s2v.'), 'wb'))

    def _generate_s2v(self, sentence):
        sent1_s2v, _, _ = generate_s2v(sentence)
        return sent1_s2v

    def _init_batch(self):
        global batch_train_data, batch_valid_data, batch_full_data, batch_full_mask_data
        global batch_train_data_size, batch_valid_data_size
        if not self.valid:
            if len(self.batch_files) == 0:
                for file in os.listdir(self.batch_dir):
                    if (file.split(".")[-1] == 'pickle') and (not 'masked' in file):
                        self.batch_files.append(file.replace("_articles_wEmb.pickle", ""))
                random.shuffle(self.batch_files)

            batch_full_data = {}
            batch_full_mask_data = {}
            batch_train_data = []
            batch_valid_data = []
            num_of_files_in_epoch = 10*4
            for _ in range(num_of_files_in_epoch):
                file = self.batch_files[self.batch_idx]
                data = pickle.load(open(self.batch_dir+file+"_articles_wEmb.pickle", 'rb'))
                # data_masked = pickle.load(open(self.batch_dir+file+"_masked_articles_wEmb.pickle", 'rb'))
                valid_cnt = 0 # first two document in batch file is mark always as test
                for doc in data:
                    if valid_cnt < 2:
                        batch_valid_data.append(doc)
                        valid_cnt += 1
                    else:
                        batch_train_data.append(doc)
                    batch_full_data[doc] = data[doc]
                    # batch_full_mask_data[doc] = data_masked[doc]
                self.batch_idx += 1
                if self.batch_idx >= len(self.batch_files):
                    self.batch_idx = 0

            batch_train_data_size = 0
            for title in batch_train_data:
                batch_train_data_size += len(batch_full_data[title])
            batch_valid_data_size = 0
            for title in batch_valid_data:
                batch_valid_data_size += len(batch_full_data[title])
            print("\tTrain dataset size: " + str(batch_train_data_size))
            print("\tTest dataset size: " + str(batch_valid_data_size))

            global train_title_list, valid_title_list
            global train_title_weight, valid_title_weight
            train_title_list = []
            train_title_weight = []
            for title in batch_train_data:
                train_title_list.append(title)
                train_title_weight.append(len(batch_full_data[title]))
            valid_title_list = []
            valid_title_weight = []
            for title in batch_valid_data:
                valid_title_list.append(title)
                valid_title_weight.append(len(batch_full_data[title]))
            
            if self.config['use_memory'] and 'closest' in self.config['training']['memory_sentence_pos']:
                self.find_closest_s2vs()

    def find_closest_s2vs(self):
        global batch_full_data
        for title in batch_full_data:
            s2vs_map = {}
            for sidx in range(len(batch_full_data[title])):
                sent_s2v = np.mean(batch_full_data[title][sidx]['sentence_emb'][0], axis=0)
                s2vs_map[sidx] = sent_s2v
            for sidx in s2vs_map:
                best_dist = 1e6
                best_idx = 0
                for sidx2 in np.random.randint(0, len(s2vs_map)-1,
                    size=(min(len(s2vs_map), int(self.config['training']['memory_sentence_pos'].replace('closest', ''))))):
                # for sidx2 in s2vs_map:
                    if sidx2 != sidx:
                        dist = np.sum(np.abs(s2vs_map[sidx]-s2vs_map[sidx2]))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = sidx2
                batch_full_data[title][sidx]['closest_s2v'] = best_idx

    def on_epoch_end(self):
        self.true_s2v_rate *= self.config['training']['memory_true_s2v_gamma']
        print('true_s2v_rate:', self.true_s2v_rate)
        self._init_batch()

    def __len__(self):
        if self.valid:
            return int(batch_valid_data_size)
        else:
            return int(batch_train_data_size//4)

    def get_title_from_idx(self):
        global train_title_list, valid_title_list
        global train_title_weight, valid_title_weight
        title_list = valid_title_list if self.valid else train_title_list
        title_weight = valid_title_weight if self.valid else train_title_weight
        rnd_title = random.choices(title_list, weights=title_weight)[0]
        return rnd_title

    def __getitem__(self, idx):
        global batch_train_data, batch_valid_data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.config['num_mem_sents'] = 3

        mem_sentence = torch.zeros((self.config['num_mem_sents'], self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        mem_sentence_mask = torch.ones((self.config['num_mem_sents'], self.config['max_sent_len'],), dtype=torch.bool)

        masked_sentence = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        masked_sentence_mask = torch.ones((self.config['max_sent_len'],), dtype=torch.bool)

        label_sentence = torch.zeros((self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        label_sentence_mask = torch.ones((self.config['max_sent_len'],), dtype=torch.bool)

        sent_order = torch.zeros((1,), dtype=torch.long)

        title = self.get_title_from_idx()
        batch_data = batch_full_data[title]

        # memory sentence
        # rnd_mem_sent_idx = random.randint(0, len(batch_data) - 1 - self.config['num_mem_sents'])
        # for mem_idx in range(0, self.config['num_mem_sents']):
        #     sent = batch_data[rnd_mem_sent_idx+mem_idx]['sentence_emb'][0]
        #     if self.config['use_memory']:
        #         mem_sentence[mem_idx][0:min(len(sent), self.config['max_sent_len'])] =\
        #             torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        #     mem_sentence_mask[mem_idx][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

        rnd_mem_sent_idx = random.randint(0, len(batch_data) - 1 - 2)
        sent = batch_data[rnd_mem_sent_idx]['sentence_emb'][0]
        if self.config['use_memory']:
            mem_sentence[0][0:min(len(sent), self.config['max_sent_len'])] =\
                torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        mem_sentence_mask[0][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

        rnd_order = random.choice([False, True])
        rnd_order = True # <-----------------------------------
        sent_order[0] = rnd_order
        if rnd_order:
            sent = batch_data[rnd_mem_sent_idx+1]['sentence_emb'][0]
        else:
            rnd_mem_sent_idx = random.randint(0, len(batch_data) - 1 - 1)
            sent = batch_data[rnd_mem_sent_idx]['sentence_emb'][0]

        if self.config['use_memory']:
            mem_sentence[1][0:min(len(sent), self.config['max_sent_len'])] =\
                torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        mem_sentence_mask[1][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

        # 2nd sent from other document
        title2 = title
        while title2 == title:
            title2 = self.get_title_from_idx()
        batch_data2 = batch_full_data[title2]
        rnd_doc2_mem_sent_idx = random.randint(0, len(batch_data2) - 1)
        sent = batch_data2[rnd_doc2_mem_sent_idx]['sentence_emb'][0]
        if self.config['use_memory']:
            mem_sentence[2][0:min(len(sent), self.config['max_sent_len'])] =\
                torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        mem_sentence_mask[2][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

        # masked sentence
        if self.config['training']['memory_sentence_pos'] == "rnd":
            rnd_input_sent_idx = random.randint(0, len(batch_data)-2)
        elif self.config['training']['memory_sentence_pos'] == "-1":
            rnd_input_sent_idx = rnd_mem_sent_idx + 2
        # rnd_input_sent_idx = rnd_mem_sent_idx
        # rnd_input_sent_idx = rnd_mem_sent_idx + self.config['num_mem_sents']
        # if rnd_input_sent_idx in range(rnd_mem_sent_idx, rnd_mem_sent_idx+self.config['num_mem_sents']):
        #     rnd_input_sent_idx = len(batch_data)-1
        sent = batch_data[rnd_input_sent_idx]['sentence_emb'][0]
        masked_sentence[0:min(len(sent), self.config['max_sent_len'])] =\
            torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        masked_sentence_mask[0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)
        if self.config['training']['input_drop'] != 0.0:
            # drop_mask = torch.from_numpy(
            #     np.random.binomial(1, 1-self.config['training']['input_drop'],
            #                        (self.config['max_sent_len'], self.config['word_edim'])
            #                       ).astype(np.float32))
            drop_mask = torch.from_numpy(
                np.random.binomial(1, 1-self.config['training']['input_drop'],
                                   (self.config['word_edim'])
                                  ).astype(np.float32))
            masked_sentence = masked_sentence*drop_mask

        # label sentence
        sent = batch_data[rnd_input_sent_idx]['sentence_emb'][0]
        # label_sentence[0:min(len(sent), self.config['max_sent_len'])] =\
        #     torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        # label_sentence_mask[0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)
        label_sentence[1:min(len(sent)+1, self.config['max_sent_len'])] =\
            torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len']-1)].astype(np.float32))
        label_sentence_mask[1:min(len(sent)+1, self.config['max_sent_len'])] = torch.tensor(0.0)

        return masked_sentence, masked_sentence_mask, mem_sentence, mem_sentence_mask, \
               label_sentence, label_sentence_mask, sent_order

    def get_sentence(self, title_idx, sent_idx):
        global batch_train_data, batch_valid_data
        sentence = torch.zeros((1, self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
        sentence_mask = torch.ones((1, self.config['max_sent_len'],), dtype=torch.bool)

        title_list = valid_title_list if self.valid else train_title_list
        title = title_list[title_idx]
        batch_data = batch_full_data[title]

        print(title)

        sent = batch_data[sent_idx]['sentence_emb'][0]
        sentence[0][0:min(len(sent), self.config['max_sent_len'])] =\
            torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
        sentence_mask[0][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

        print(batch_data[sent_idx]['sentence'])
        
        return sentence, sentence_mask

    def get_sentences_from_doc(self, title_idx):
        global batch_train_data, batch_valid_data
        title_list = valid_title_list if self.valid else train_title_list
        title = title_list[title_idx]
        batch_data = batch_full_data[title]

        sentences = []
        for sent_idx in range(0, len(batch_data)-1):
            sentence = torch.zeros((1, 3, self.config['max_sent_len'], self.config['word_edim'],), dtype=torch.float)
            sentence_mask = torch.ones((1, 3, self.config['max_sent_len'],), dtype=torch.bool)

            sent = batch_data[sent_idx]['sentence_emb'][0]
            for i in range(0, 3):
                sentence[0][i][0:min(len(sent), self.config['max_sent_len'])] =\
                    torch.from_numpy(sent[0:min(len(sent), self.config['max_sent_len'])].astype(np.float32))
                sentence_mask[0][i][0:min(len(sent), self.config['max_sent_len'])] = torch.tensor(0.0)

            sentences.append({'idx': sent_idx, 'text': batch_data[sent_idx]['sentence'],
                              'emb': sentence, 'mask': sentence_mask})
        
        return sentences

def test():
    batcher = WikiS2vCorrectionBatch({
        's2v_dim': 4096,
        'doc_len': 400
    })
    for i in range(100):
        x = batcher.__getitem__(i)
        # print(x)


# test()
