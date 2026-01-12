import torch
from rouge_score import rouge_scorer
from tqdm import tqdm

def evaluate_model(model, batch, tokenizer, device):
    model.eval()

    # Инициализируем scorer для ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=False)
    
    batch_rouge1 = 0
    batch_rouge2 = 0

    # Вычисляем ROUGE метрики для батча
    with torch.no_grad():
       
        # Перемещаем данные на устройство
        x_batch = batch['context'].to(device)
        y_batch = batch['target'].to(device)

        logits, _ = model(x_batch)
        
        # Получаем предсказания (greedy decoding)
        predictions = torch.argmax(logits, dim=-1)
        batch_size = x_batch.size(0)
        
        for i in range(batch_size):
            
            # Извлекаем целевые токены
            target_tokens = [tokenizer.decode(id) for id in y_batch]
            
            # Извлекаем предсказанные токены
            pred_tokens = [tokenizer.decode(id) for id in predictions]
            
            # Преобразуем в строки для ROUGE
            target_text = ' '.join(target_tokens)
            pred_text = ' '.join(pred_tokens)
            
            # Вычисляем ROUGE
            if target_text and pred_text:
                scores = scorer.score(target_text, pred_text)
                batch_rouge1 += scores['rouge1'].fmeasure
                batch_rouge2 += scores['rouge2'].fmeasure
        
        # Усредняем по батчу
        if batch_size > 0:
            avg_rouge1 = batch_rouge1 / batch_size
            avg_rouge2 = batch_rouge2 / batch_size

    return avg_rouge1, avg_rouge2