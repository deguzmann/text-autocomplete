import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from src.eval_lstm import evaluate_model


def train_sequence(model, dataloader, criterion, optimizer, 
                                    tokenizer, device, clip_norm=1.0):
    """
    Одна эпоха обучения для последовательностей с метриками ROUGE
    
    Args:
        model: LSTM модель
        dataloader: DataLoader с данными
        criterion: функция потерь (CrossEntropyLoss)
        optimizer: оптимизатор
        tokenizer: обратный словарь {индекс: слово}
        device: устройство (cpu/mps/cuda)
        clip_norm: значение для gradient clipping
    
    Returns:
        avg_loss: средний loss за эпоху
        rouge1_f1: средний F1-score ROUGE-1
        rouge2_f1: средний F1-score ROUGE-2
    """
    model.train()
    total_loss = 0
    total_batches = 0
    total_rouge1_f1 = 0
    total_rouge2_f1 = 0
        
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # История для графиков
    history = {"train_loss": [], "val_rouge1": [], "val_rouge2": [], "batch_cnt": []}
    
    for batch_idx, item in enumerate(progress_bar):
        # Перемещаем данные на устройство
        x_batch = item['context'].to(device)
        y_batch = item['target'].to(device)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Forward pass - получаем логиты для ВСЕХ позиций
        logits, _ = model(x_batch)
        
        # Reshape для CrossEntropyLoss
        logits_flat = logits.reshape(-1, logits.size(-1))
        y_flat = y_batch.reshape(-1)
        
        # Вычисляем loss (ignore_index уже учтен в criterion)
        loss = criterion(logits_flat, y_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        
        # Шаг оптимизатора
        optimizer.step()

        # Обновляем общий loss
        total_loss += loss.item()
        
        batch_size = x_batch.size(0)
        avg_rouge1, avg_rouge2 = evaluate_model(model, item, tokenizer, device)
        model.train()
        
        total_batches += 1
        total_rouge1_f1 += avg_rouge1
        total_rouge2_f1 += avg_rouge2

        # Обновляем progress bar
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rouge1': f"{avg_rouge1:.4f}" if batch_size > 0 else "0.0000",
                'rouge2': f"{avg_rouge2:.4f}" if batch_size > 0 else "0.0000",
                'device': device
            })

            history['train_loss'].append(loss.item())
            history['val_rouge1'].append(avg_rouge1)
            history['val_rouge2'].append(avg_rouge2)
            history['batch_cnt'].append(total_batches)

            torch.save(model, './models/full_model.pth') # сохранение всей модели
            torch.save(model.state_dict(), './models/model_weights.pth') # сохранение весов

    
    # Усредняем метрики по всем батчам
    avg_loss = total_loss / len(dataloader)
    avg_rouge1 = total_rouge1_f1 / total_batches if total_batches > 0 else 0
    avg_rouge2 = total_rouge2_f1 / total_batches if total_batches > 0 else 0
    
    return avg_loss, avg_rouge1, avg_rouge2, history