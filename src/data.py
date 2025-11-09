"""数据处理模块"""
import spacy
import torch
from datasets import load_dataset
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


def load_tokenizers():
    try:
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        print("未找到 'en_core_web_sm'。请运行: python -m spacy download en_core_web_sm")
        exit()
    
    try:
        spacy_de = spacy.load('de_core_news_sm')
    except IOError:
        print("未找到 'de_core_news_sm'。请运行: python -m spacy download de_core_news_sm")
        exit()
    
    return spacy_en, spacy_de


def tokenize_en(text, spacy_en):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text, spacy_de):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, language):
    for data_sample in data_iter:
        if language == 'en':
            yield from tokenizer(data_sample['translation']['en'])
        else:
            yield from tokenizer(data_sample['translation']['de'])


def load_dataset_iwslt(train_samples):
    print("正在加载 IWSLT2017 数据集...")
    dataset = load_dataset("iwslt2017", "iwslt2017-de-en")
    train_data = dataset["train"].select(range(train_samples))
    valid_data = dataset["validation"]
    test_data = dataset["test"]
    
    print(f"训练集样本数: {len(train_data)}")
    print(f"验证集样本数: {len(valid_data)}")
    print(f"测试集样本数: {len(test_data)}")
    
    return train_data, valid_data, test_data


def build_vocab(train_data, spacy_en, spacy_de):
    print("正在构建词汇表 (使用 Counter+Vocab)...")
    
    def tokenize_en_wrapper(data_sample):
        return tokenize_en(data_sample['translation']['en'], spacy_en)
    
    def tokenize_de_wrapper(data_sample):
        return tokenize_de(data_sample['translation']['de'], spacy_de)
    
    def yield_en_tokens():
        for data_sample in train_data:
            yield from tokenize_en_wrapper(data_sample)
    
    def yield_de_tokens():
        for data_sample in train_data:
            yield from tokenize_de_wrapper(data_sample)
    
    counter_src = Counter(yield_en_tokens())
    vocab_src = Vocab(counter_src, min_freq=1, specials=special_symbols)
    
    counter_tgt = Counter(yield_de_tokens())
    vocab_tgt = Vocab(counter_tgt, min_freq=1, specials=special_symbols)
    
    print("--- 正在验证特殊标记索引 ---")
    print(f"  <unk> 索引 (应为 0): {vocab_src['<unk>']}")
    print(f"  <pad> 索引 (应为 1): {vocab_src['<pad>']}")
    print(f"  <bos> 索引 (应为 2): {vocab_src['<bos>']}")
    print(f"  <eos> 索引 (应为 3): {vocab_src['<eos>']}")
    
    if vocab_src['<pad>'] != PAD_IDX:
        print("\n[!!! 严重错误 !!!]: PAD 索引不匹配!")
        print(f"我们的代码期望 PAD_IDX = {PAD_IDX}, 但词汇表中的 <pad> = {vocab_src['<pad>']}")
        print("请停止程序并检查!")
    else:
        print("--- 索引验证通过 ---")
    
    print(f"源 (EN) 词汇表大小: {len(vocab_src)}")
    print(f"目标 (DE) 词汇表大小: {len(vocab_tgt)}")
    
    return vocab_src, vocab_tgt


def create_collate_fn(vocab_src, vocab_tgt, spacy_en, spacy_de, device):
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        
        for data_sample in batch:
            tokens_en = tokenize_en(data_sample['translation']['en'], spacy_en)
            indices_en = [vocab_src[token] for token in tokens_en]
            src_tensor = torch.tensor(indices_en, dtype=torch.long)
            
            tokens_de = tokenize_de(data_sample['translation']['de'], spacy_de)
            indices_de = [vocab_tgt[token] for token in tokens_de]
            tgt_tensor = torch.tensor(indices_de, dtype=torch.long)

            src_with_bos_eos = torch.cat([torch.tensor([BOS_IDX]), src_tensor, torch.tensor([EOS_IDX])])
            tgt_with_bos_eos = torch.cat([torch.tensor([BOS_IDX]), tgt_tensor, torch.tensor([EOS_IDX])])
            
            src_batch.append(src_with_bos_eos)
            tgt_batch.append(tgt_with_bos_eos)

        src_batch_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
        tgt_batch_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)

        dec_inputs = tgt_batch_padded[:, :-1]
        dec_outputs = tgt_batch_padded[:, 1:]

        return src_batch_padded.to(device), dec_inputs.to(device), dec_outputs.to(device)
    
    return collate_fn


def create_dataloaders(train_data, valid_data, batch_size, collate_fn):
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return loader, valid_loader

