import random
import numpy as np
import os
import sys
from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm
import torch


class DataLoader(object):
    def __init__(self, data_dir, params):

        # loading vocab (we require this to map words to their indices)
        self.params = params
        self.data_dir = data_dir

        self.vocab_w2v = {}
        self.weights_w2v = []
        self.index_to_word_w2v = {}

        vocab_path = os.path.join(data_dir, 'fold' + str(params.fold), 'vectors.kv')
        word_vectors = KeyedVectors.load(vocab_path)

        for idx, key in enumerate(word_vectors.vocab):
            self.vocab_w2v[key] = idx
            self.weights_w2v.append(word_vectors[key])
            self.index_to_word_w2v[idx] = key
        self.vocab_w2v['<UNK>'] = idx + 1
        self.unk_ind_w2v = idx + 1
        self.index_to_word_w2v[idx + 1] = '<UNK>'
        vec = np.random.randn(100)
        vec = vec / float(np.linalg.norm(vec) + 1e-6)
        self.weights_w2v.append(vec.astype(np.float32))
        self.vocab_w2v['<PAD>'] = idx + 2
        self.pad_ind_w2v = idx + 2
        self.index_to_word_w2v[idx + 2] = '<PAD>'
        self.weights_w2v.append(np.zeros(100, dtype=np.float32))
        self.weights_w2v = np.stack(self.weights_w2v, axis=0)

        # adding dataset parameters to param (e.g. vocab size, )
        params.vocab_size_w2v = len(self.vocab_w2v)

    def pad_or_truncate(self, note, pad_ind):
        if (len(note) > self.params.doc_length):
            return note[-self.params.doc_length:], [0.0] * self.params.doc_length, [False] * self.params.doc_length
        else:
            attn_mask = [0.0] * len(note) + [float('-inf')] * (self.params.doc_length - len(note))
            bool_attn_mask = [False] * len(note) + [True] * (self.params.doc_length - len(note))
            note.extend([pad_ind] * (self.params.doc_length - len(note)))
            return note, attn_mask, bool_attn_mask

    def load_notes_labels(self, file, d):
        df_notes = pd.read_csv(file)

        if (self.params.debug):
            df_notes = df_notes.head(self.params.batch_size * 2)
            self.params.icd_threshold = 1

        total = len(df_notes.index)
        notes_w2v = []
        w2v_attn_mask = []
        w2v_bool_attn_mask = []
        labels = []
        icds = []
        ids = []
        df_notes.dropna(axis=1, inplace=True)
        if 'train' in file:
            if self.params.phenotype == 'diag_only':
                self.icd_cols = [x for x in df_notes.columns if
                                 'ROLLED_ICD_DIAG_' in x and df_notes[x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'proc_only':
                self.icd_cols = [x for x in df_notes.columns if
                                 'ROLLED_ICD_PROC_' in x and df_notes[x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'phec_only':
                self.icd_cols = [x for x in df_notes.columns if
                                 'ROLLED_PHEC' in x and df_notes[x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'rolled':
                self.icd_cols = [x for x in df_notes.columns if
                                 'ROLLED_ICD' in x and df_notes[x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'full':
                self.icd_cols = [x for x in df_notes.columns if
                                 ('ICD_PROC' in x or 'ICD_DIAG' in x) and 'ROLLED' not in x and df_notes[
                                     x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'rolled_phecodes':
                self.icd_cols = [x for x in df_notes.columns if
                                 ('ROLLED_ICD_PROC_' in x or 'ROLLED_PHEC' in x) and df_notes[
                                     x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'phecodes':
                self.icd_cols = [x for x in df_notes.columns if
                                 ('ROLLED_ICD_PROC_' in x or ('PHEC' in x and 'ROLLED' not in x)) and df_notes[
                                     x].sum() >= self.params.icd_threshold]
            elif self.params.phenotype == 'ccs':
                self.icd_cols = [x for x in df_notes.columns if
                                 'CCS' in x and df_notes[x].sum() >= self.params.icd_threshold]
            d['num_phenotypes'] = len(self.icd_cols)
            d['icd_cols'] = self.icd_cols
        with tqdm(total=total) as pbar:
            for index, row in tqdm(df_notes.iterrows()):
                pbar.update(1)
                text = row['TEXT']
                icd = [row[x] for x in self.icd_cols]
                note_w2v = [self.vocab_w2v[token] if token in self.vocab_w2v else self.vocab_w2v['<UNK>'] for token in
                            text.split()]
                if 'bounceback' in self.params.task:
                    label = [row['IsReadmitted_Bounceback']]
                elif '30_day' in self.params.task:
                    label = [row['IsReadmitted_30days']]
                elif 'mortality30' in self.params.task:
                    label = [row['Mortality_30days']]
                elif 'mortality_in' in self.params.task:
                    label = [row['Mortality_InHospital']]
                elif self.params.task == 'icd_only':
                    label = [0]
                else:
                    print('Invalid task: ' + str(self.params.task))
                    sys.exit()
                id = row['ICUSTAY_ID']

                note_w2v_ind, note_w2v_mask, note_bool_w2v_mask = self.pad_or_truncate(note_w2v, self.pad_ind_w2v)
                notes_w2v.append(note_w2v_ind)
                w2v_attn_mask.append(note_w2v_mask)
                w2v_bool_attn_mask.append(note_bool_w2v_mask)
                labels.append(label)
                ids.append(id)
                icds.append(icd)

        # checks to ensure there is a label for each note
        assert len(labels) == len(notes_w2v)
        assert len(notes_w2v) == len(w2v_attn_mask)
        assert len(notes_w2v) == len(w2v_bool_attn_mask)
        assert len(ids) == len(icds)

        # store note data in dict d
        d['data_w2v'] = notes_w2v
        d['mask_w2v'] = w2v_attn_mask
        d['bool_mask_w2v'] = w2v_bool_attn_mask
        d['labels'] = labels
        d['size'] = len(notes_w2v)
        d['ids'] = ids
        d['icds'] = icds
        d['num_phenotypes'] = len(self.icd_cols)
        d['icd_cols'] = self.icd_cols

    def load_data(self, types, data_dir):
        """
        Loads the data for all training splits and stores in a dictionary
        """
        data = {}

        for split in ['train', 'val', 'test']:
            if split in types:
                file = os.path.join(data_dir, 'fold' + str(self.params.fold), 'df_' + split + '_subjects.csv')
                data[split] = {}
                self.load_notes_labels(file, data[split])
        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Data generator that yields the data needed for each batch
        """

        order = list(range(data['size']))
        if shuffle:
            random.shuffle(order)

        # one pass over data
        for i in range((data['size']) // params.batch_size):
            # fetch data
            batch_notes_w2v = np.array(
                [data['data_w2v'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_w2v_mask = np.array(
                [data['mask_w2v'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_bool_w2v_mask = np.array(
                [data['bool_mask_w2v'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_tags = np.array(
                [data['labels'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])
            batch_icd = np.array(
                [data['icds'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]])

            batch_ids = [data['ids'][idx] for idx in order[i * params.batch_size:(i + 1) * params.batch_size]]

            batch_data_w2v, batch_labels = torch.tensor(batch_notes_w2v, dtype=torch.long, device='cuda'), torch.tensor(
                batch_tags, dtype=torch.float, device='cuda')
            batch_icd = torch.tensor(batch_icd, dtype=torch.float, device='cuda')

            if 'attn' in self.params.model:
                batch_w2v_mask = torch.tensor(batch_w2v_mask, dtype=torch.float, device='cuda')

                batch_bool_w2v_mask = torch.tensor(batch_bool_w2v_mask, dtype=torch.bool, device='cuda')
                if self.params.task == 'icd_only':
                    yield [batch_data_w2v, batch_w2v_mask], batch_icd, batch_ids
                elif 'frozen' in self.params.task:
                    yield [batch_data_w2v, batch_w2v_mask, batch_bool_w2v_mask], batch_labels, batch_ids
                else:
                    yield [batch_data_w2v, batch_w2v_mask, batch_bool_w2v_mask], batch_labels, batch_icd, batch_ids
            elif 'lr' in self.params.model:
                yield batch_icd, batch_labels
            else:
                yield batch_data_w2v, batch_labels, batch_ids
