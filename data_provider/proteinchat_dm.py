# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import random
import torch
import os
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from data_provider.gal_helpers import escape_custom_split_sequence
from pathlib import Path
from torch.utils.data.dataloader import default_collate


class ProteinChatCollater(object):
    def __init__(self, tokenizer, q_max_len, a_max_len, use_gal):
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.use_gal = use_gal
        
    def __call__(self, batch):
        embeds, prot_seqs, questions, answers, q_types = zip(*batch)
        max_embed_len = 896
        ## concate 
        if False:
            max_dim = max([e.shape[0] for e in embeds])

            padded_embeds = []
            for embed in embeds:
                shape_dim0 = embed.shape[0]
                pad1 = ((0, max_dim - shape_dim0), (0, 0), (0, 0))
                padded_embeds.append(np.pad(embed, pad1, mode='constant'))
            padded_embeds = default_collate(padded_embeds).squeeze(dim=2)[:,:1024,:]
        else:
            padded_embeds = torch.zeros(len(embeds), max_embed_len, 512)
            for i in range(len(embeds)):
                padded_embeds[i, :embeds[i].shape[0], :] = embeds[i][:max_embed_len, :]
            padded_embeds = padded_embeds.detach()

        assert len(prot_seqs) == len(questions) == len(answers)

        if self.use_gal:
            questions = [escape_custom_split_sequence(q) for q in questions]
        answers = [a + '\n' for a in answers]
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
        prot_mask = torch.ones(padded_embeds.shape[0], padded_embeds.shape[1], dtype=torch.bool)
        return (padded_embeds, prot_mask), q_batch, a_batch


class InferenceCollater(object):
    def __init__(self, tokenizer, q_max_len, a_max_len, use_gal):
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.use_gal = use_gal
        
    def __call__(self, batch):
        embeds, prot_seqs, questions, answers, q_types = zip(*batch)
        max_embed_len = 896
        ## concate 
        if False:
            max_dim = max([e.shape[0] for e in embeds])

            padded_embeds = []
            for embed in embeds:
                shape_dim0 = embed.shape[0]
                pad1 = ((0, max_dim - shape_dim0), (0, 0), (0, 0))
                padded_embeds.append(np.pad(embed, pad1, mode='constant'))
            padded_embeds = default_collate(padded_embeds).squeeze(dim=2)[:,:1024,:]
        else:
            padded_embeds = torch.zeros(len(embeds), max_embed_len, 512)
            for i in range(len(embeds)):
                padded_embeds[i, :embeds[i].shape[0], :] = embeds[i][:max_embed_len, :]
            padded_embeds = padded_embeds.detach()

        assert len(prot_seqs) == len(questions) == len(answers)

        if self.use_gal:
            questions = [escape_custom_split_sequence(q) for q in questions]
        answers = [a + '\n' for a in answers]
        self.tokenizer.padding_side = 'left'
        q_batch = self.tokenizer(questions,
                                 truncation=True,
                                 padding='max_length',
                                 add_special_tokens=True,
                                 max_length=self.q_max_len,
                                 return_tensors='pt',
                                 return_attention_mask=True, 
                                 return_token_type_ids=False)
        prot_mask = torch.ones(padded_embeds.shape[0], padded_embeds.shape[1], dtype=torch.bool)
        target_dict = {'answers': answers, "q_types": q_types}
        return (padded_embeds, prot_mask), q_batch, target_dict


class ProteinChatDM(LightningDataModule):
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
        self.prompt = args.prompt
        
        self.train_dataset = ProteinChatDataset(root, 'train.txt', prompt="### Human: {}\n### Assistant: ", pt_file_path=args.pt_file_path)
        self.val_dataset = ProteinChatDataset(root, 'val.txt', prompt="### Human: {}\n### Assistant: ", pt_file_path=args.pt_file_path)
        self.test_dataset = ProteinChatDataset(root, 'test.txt', prompt="### Human: {}\n### Assistant: ", pt_file_path=args.pt_file_path)
        
        self.tokenizer = None
        self.use_gal = args.llm_name.find('gal') >= 0
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=ProteinChatCollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal),
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
            collate_fn=ProteinChatCollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=InferenceCollater(self.tokenizer, self.q_max_len, self.a_max_len, self.use_gal),
        )
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--q_max_len', type=int, default=30)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prompt', type=str, default='[START_AMINO]{}[END_AMINO]. Question: {} Answer:')
        parser.add_argument('--pt_file_path', type=str, default='/home/XXXX-2/proteinchatdata/proteinchat')
        return parent_parser



class ProteinChatDataset(Dataset):
    def __init__(self, root_path, subset, pt_file_path, prompt="Question: {} Answer:"):
        super(ProteinChatDataset, self).__init__()
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
        
        ## process dataset
        pdb_set = set(i[0] for i in pdb2seq)
        ## filter qa data
        qa_data = {k: v for k, v in qa_data.items() if k in pdb_set}
        assert len(qa_data) == len(pdb_set), print(len(qa_data), len(pdb_set))
        
        pt_file = Path(pt_file_path).glob('*.pt')
        pt_file_ids = {f.name.split('.pt')[0] for f in pt_file}
        self.pt_file_path = pt_file_path

        ## load q types
        with open(self.q_type_path, 'r') as f:
            q_types = [line.strip().split('\t') for line in f.readlines()]
        self.q_type_dict = {q: t for q, t in q_types}

        ## generate qa data
        self.data_list = []
        for pdb_id, seq in pdb2seq:
            if pdb_id not in pt_file_ids:
                continue
            qa_list = qa_data[pdb_id]
            for qa in qa_list:
                q = qa['Q']
                a = str(qa['A'])
                self.data_list.append((pdb_id, seq, q, a))

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pdb_id, seq, q, a = self.data_list[index]
        q_type = self.q_type_dict[q]
        path = os.path.join(self.pt_file_path, pdb_id + '.pt')
        embed = torch.load(path, map_location=torch.device('cpu'))
        embed = embed.squeeze(dim=1)
        embed = embed.detach()
        q = self.prompt.format(q)
        return embed, seq, q, a, q_type


if __name__ == '__main__':
    dataset = ProteinChatDataset('./data/PDBDataset', 'train.txt')
    dataset.shuffle()
    for i in range(1000):
        print(dataset[i][0].shape)
    