import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import json


# класс датасета
class TweetDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=20):
        self.samples = []
        self.seq_len = seq_len
        
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=False, max_length=512, truncation=True)
            
            n_tokens = len(token_ids)
            
            if n_tokens <= seq_len:
                # Короткий твит: используем весь как контекст
                # context: все токены кроме последнего, target: все токены кроме первого
                if n_tokens > 1:
                    context = token_ids[:-1]
                    target = token_ids[1:]
                    self.samples.append((context,target))
            else:
                # Длинный твит: скользящее окно
                for i in range(0, n_tokens - seq_len):
                    context = token_ids[i:seq_len+i]
                    target = token_ids[i+1:seq_len+i+1]
                    self.samples.append((context, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        
        return {
            'context': torch.tensor(x, dtype=torch.long),
            'target': torch.tensor(y, dtype=torch.long),
        }
    
    def save_vocab_json(vocab, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    
def collate_fn(batch, tokenizer): 
    # список текстов и классов из батча
    contexts = [item['context'] for item in batch]
    targets = [item['target'] for item in batch]

    if tokenizer is None:  # ← ДОБАВЬТЕ ПРОВЕРКУ
        raise ValueError("Tokenizer is None!")
    
    # дополняем тексты в батче padding'ом
    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=tokenizer.eos_token_id)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=tokenizer.eos_token_id)
    # возвращаем преобразованный батч
    return {
        'context': padded_contexts,
        'target': padded_targets,
    }
