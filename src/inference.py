
import torch
from data import BOS_IDX, EOS_IDX, tokenize_en, tokenize_de


def greedy_decoder(model, enc_input, start_symbol, device):
    model.eval()
    
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    
    dec_input = torch.zeros(1, 1).type_as(enc_input.data).fill_(start_symbol)
    
    terminal = False
    next_symbol = start_symbol
    output_tokens = []

    while not terminal:
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected[:, -1, :].squeeze(0)
        _, next_word_idx = torch.max(prob, dim=-1)
        next_symbol = next_word_idx.item()
        
        output_tokens.append(next_symbol)

        if next_symbol == EOS_IDX or len(output_tokens) > 50:
            terminal = True

        dec_input = torch.cat([dec_input.to(device),
                               torch.tensor([[next_symbol]], device=device)], dim=1)
    
    return output_tokens[:-1]


def test_translation(model, test_data, vocab_src, vocab_tgt, spacy_en, device):
    model.eval()
    
    test_iter = iter(test_data)
    test_sample = next(test_iter)
    src_text = test_sample['translation']['en']
    tgt_text = test_sample['translation']['de']
    
    print("\n" + "="*30)
    print("开始测试翻译...")
    print(f"源 (EN): {src_text}")
    print(f"目标 (DE): {tgt_text}")
    
    tokens_en = tokenize_en(src_text, spacy_en)
    indices_en = [vocab_src[token] for token in tokens_en]
    src_tensor = torch.tensor(indices_en, dtype=torch.long)
    
    src_with_bos_eos = torch.cat([torch.tensor([BOS_IDX]), src_tensor, torch.tensor([EOS_IDX])])
    enc_input = src_with_bos_eos.unsqueeze(0).to(device)
    
    greedy_dec_predict_idx = greedy_decoder(model, enc_input, start_symbol=BOS_IDX, device=device)
    greedy_dec_predict_words = [vocab_tgt.itos[idx] for idx in greedy_dec_predict_idx]
    
    print(f"模型翻译 (DE): {' '.join(greedy_dec_predict_words)}")
    print("="*30)

