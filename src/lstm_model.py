import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, dropout=0.3, rnn_type="LSTM"):
        super().__init__()

        # 1. Слой эмбеддингов
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=0
        )

        # 2. LSTM слои
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. Дропаут
        self.dropout = nn.Dropout(dropout)

        # 4. Выходной слой
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        # Обычный LSTM
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)

        output = self.dropout(output)

        linear_out = self.fc(output)

        return linear_out, hidden

    def predict_sequence(self, x, tokenizer, max_length=50, temperature=1.0):

        # Гарантируем batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # (1, seq_len)
        
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            input_tokens = x.to(device)
            _, hidden = self.forward(input_tokens)
            
            current_input = input_tokens
            predictions = current_input
            
            for _ in range(max_length + 1):
                
                logits, hidden = self.forward(current_input, hidden)
                last_logits = logits[:,-1, :] / temperature
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # predictions.append(next_token)
                predictions = torch.cat([predictions,next_token], dim=1)

                # Можно остановиться на EOS токене
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        return predictions
    

