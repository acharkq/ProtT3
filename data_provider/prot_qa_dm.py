# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
from pytorch_lightning import LightningDataModule
# import torch_geometric
from torch.utils.data import DataLoader, Dataset
from pathlib import Path


class ProtQACollater(object):
    def __init__(self, tokenizer, prot_tokenizer, q_max_len, a_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.prot_max_len = prot_max_len
        
    def __call__(self, batch):
        prot_seqs, questions, answers, _, _ = zip(*batch)
        answers = [a + '\n' for a in answers]
        prot_batch = self.prot_tokenizer(prot_seqs,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=self.prot_max_len,
                                         return_tensors="pt",
                                         return_attention_mask=True, 
                                         return_token_type_ids=False)
        if False:
            self.tokenizer.padding_side = 'left'
            q_batch = self.tokenizer(questions,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.q_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True, 
                                    return_token_type_ids=False)
            self.tokenizer.padding_side = 'right'
            a_batch = self.tokenizer(answers,
                                    truncation=True,
                                    padding='max_length',
                                    add_special_tokens=True,
                                    max_length=self.a_max_len,
                                    return_tensors='pt',
                                    return_attention_mask=True, 
                                    return_token_type_ids=False)
            return prot_batch, q_batch, a_batch
        else:
            self.tokenizer.padding_side = 'right'
            qa_pair = [[q, a] for q, a in zip(questions, answers)]
            qa_batch = self.tokenizer(qa_pair,
                                      truncation=True,
                                      padding='max_length',
                                      add_special_tokens=True,
                                      max_length=self.q_max_len + self.a_max_len,
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      return_token_type_ids=True)
            return prot_batch, qa_batch


class InferenceCollater(object):
    def __init__(self, tokenizer, prot_tokenizer, q_max_len, a_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.prot_max_len = prot_max_len
        
    def __call__(self, batch):
        prot_seqs, questions, answers, q_types, indices = zip(*batch)
        answers = [a + '\n' for a in answers]
        prot_batch = self.prot_tokenizer(prot_seqs,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=self.prot_max_len,
                                         return_tensors="pt",
                                         return_attention_mask=True, 
                                         return_token_type_ids=False)
        self.tokenizer.padding_side = 'left'
        q_batch = self.tokenizer(questions,
                                 truncation=True,
                                 padding='max_length',
                                 add_special_tokens=True,
                                 max_length=self.q_max_len,
                                 return_tensors='pt',
                                 return_attention_mask=True, 
                                 return_token_type_ids=False)        
        target_dict = {'targets': answers, 'q_types': q_types, 'indices': indices}
        return prot_batch, q_batch, target_dict


class ProtQADM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.q_max_len = args.q_max_len
        self.a_max_len = args.a_max_len
        self.prot_max_len = args.prot_max_len
        self.prompt = args.prompt
        
        self.train_dataset = PDBQADataset(root, 'train.txt', prompt=self.prompt, filter_side_qa=args.filter_side_qa)
        self.val_dataset = PDBQADataset(root, 'val.txt', prompt=self.prompt, filter_side_qa=args.filter_side_qa)
        self.test_dataset = PDBQADataset(root, 'test.txt', prompt=self.prompt, filter_side_qa=args.filter_side_qa)
        
        self.tokenizer = None
        self.prot_tokenizer = None
    
    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=ProtQACollater(self.tokenizer, self.prot_tokenizer, self.q_max_len, self.a_max_len, self.prot_max_len),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=ProtQACollater(self.tokenizer, self.prot_tokenizer, self.q_max_len, self.a_max_len, self.prot_max_len),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.prot_tokenizer, self.q_max_len, self.a_max_len, self.prot_max_len),
        )
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--q_max_len', type=int, default=34)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        return parent_parser


class PDBQADataset(Dataset):
    def __init__(self, root_path, subset, prompt="Question: {} Answer:", filter_side_qa=False):
        super(PDBQADataset, self).__init__()
        self.data_path = Path(root_path) / subset
        self.qa_path = Path(root_path) / 'qa_all.json'
        self.q_type_path = Path(root_path) / 'q_types.txt'
        self.prompt = prompt

        ## load dataset
        with open(self.qa_path, 'r') as f:
            qa_data = json.load(f)
        
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            pdb2seq = [line.strip().split('\t') for line in lines]
        

        ## load q types
        with open(self.q_type_path, 'r') as f:
            q_types = [line.strip().split('\t') for line in f.readlines()]
        self.q_type_dict = {q: t for q, t in q_types}

        ## process dataset
        pdb_set = set(i[0] for i in pdb2seq)
        ## filter qa data
        qa_data = {k: v for k, v in qa_data.items() if k in pdb_set}
        assert len(qa_data) == len(pdb_set), print(len(qa_data), len(pdb_set))
        
        ## generate qa data
        self.data_list = []
        for pdb_id, seq in pdb2seq:
            qa_list = qa_data[pdb_id]
            for qa in qa_list:
                q = qa['Q']
                a = str(qa['A'])
                if filter_side_qa:
                    q_type = self.q_type_dict[q]
                    if q_type.find('side information') >= 0:
                        continue
                self.data_list.append((seq, q, a))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        seq, q, a = self.data_list[index]
        q_type = self.q_type_dict[q]
        q = self.prompt.format(q)
        return seq, q, a, q_type, index


if __name__ == '__main__':
    import numpy as np
    from collections import defaultdict, Counter
    train_dataset = PDBQADataset('../data/PDBDataset', 'train.txt', filter_side_qa=True)
    val_dataset = PDBQADataset('../data/PDBDataset', 'val.txt', filter_side_qa=True)
    test_dataset = PDBQADataset('../data/PDBDataset', 'test.txt', filter_side_qa=True)
    if True:
        # print(len(train_dataset), len(val_dataset), len(test_dataset))
        # train_protein_lens = np.asarray([len(p) for p in train_dataset.protein_list])
        # val_protein_lens = np.asarray([len(p) for p in val_dataset.protein_list])
        # test_protein_lens = np.asarray([len(p) for p in test_dataset.protein_list])
        
        q_lens =  []
        a_lens = []
        for seq, q, a in train_dataset.data_list:
            q_lens.append(len(q.split()))
            a_lens.append(len(a.split()))

        print(np.asarray(q_lens).min(), np.asarray(q_lens).max(), np.asarray(q_lens).mean())
        print(np.asarray(a_lens).min(), np.asarray(a_lens).max(), np.asarray(a_lens).mean())
        
        q_lens =  []
        a_lens = []
        for seq, q, a in val_dataset.data_list:
            q_lens.append(len(q.split()))
            a_lens.append(len(a.split()))

        print(np.asarray(q_lens).min(), np.asarray(q_lens).max(), np.asarray(q_lens).mean())
        print(np.asarray(a_lens).min(), np.asarray(a_lens).max(), np.asarray(a_lens).mean())

        q_lens =  []
        a_lens = []
        for seq, q, a in test_dataset.data_list:
            q_lens.append(len(q.split()))
            a_lens.append(len(a.split()))

        print(np.asarray(q_lens).min(), np.asarray(q_lens).max(), np.asarray(q_lens).mean())
        print(np.asarray(a_lens).min(), np.asarray(a_lens).max(), np.asarray(a_lens).mean())

    elif False:
        ## construct the guess for prediction by number
        train_counter = defaultdict(Counter)
        for  _, q, a in train_dataset.data_list:
            train_counter[q.lower()][a] += 1
        ## get the most common answer
        q2a = {}
        for q, counter in train_counter.items():
            q2a[q] = counter.most_common(1)[0][0]

        ## test the guess
        acc = 0
        for _, q, a in test_dataset.data_list:
            if q.lower() in q2a:
                predict = q2a[q.lower()]
                if predict.lower() == a.lower():
                    acc += 1
        print(acc / len(test_dataset.data_list))
    elif False:
        from transformers import AutoTokenizer, EsmTokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-1.3b', use_fast=False, padding_side='right')
        plm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t30_150M_UR50D')
        llm_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=ProtQACollater(llm_tokenizer, plm_tokenizer, 40, 40, 1024),
        )
    else:
        print(len(train_dataset.data_list))
        print(len(val_dataset.data_list))
        print(len(test_dataset.data_list))
        