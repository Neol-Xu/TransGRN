import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from RIEM import CAN
from util_functions import *

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=True, dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.LayerNorm(1024) 
        
        self.fc2 = nn.Linear(1024, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)  
        
        self.fc3 = nn.Linear(hidden_size, 1)
        
        self.with_dropout = with_dropout
        self.dropout_rate = dropout_rate

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.bn1(x)       
        x = F.relu(x)         
        if self.with_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)  
        
        x = self.fc2(x)
        x = self.bn2(x)          
        x = F.relu(x)            
        if self.with_dropout:
            x = F.dropout(x, p=self.dropout_rate, training=self.training) 
        
        logits = self.fc3(x)
        logits = logits.squeeze(-1).to(y.device)

        loss = F.binary_cross_entropy_with_logits(logits.double(), y.double()) 
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        acc = (preds == y.long()).sum().item() / float(y.size(0))
        return probs, loss, acc

class TransGRN(nn.Module):
    def __init__(self):
        super(TransGRN, self).__init__()
        self.can_layer = CAN(hidden_dim=384, num_heads=8, group_size=1)
        self.classifier = MLPClassifier(input_size=768 + 1536, hidden_size=256, with_dropout=True)
        
    def forward(self, TF_batch, Target_batch, batch_labels, cell_line_embeds):
        combined_features = self.can_layer(TF_batch, Target_batch)
        combined_features = torch.cat((combined_features, cell_line_embeds), dim=-1)
        logits, loss, acc = self.classifier(combined_features, batch_labels)
        
        return logits, loss, acc

def loop_dataset(data_queues, classifier, cell_line_embeddings, optimizer=None, bsize=64, mode='train'):
    total_loss = []
    all_targets = []
    all_scores = []
    queues = {k: list(v) for k, v in data_queues.items()}
    total_samples = sum(len(q) for q in queues.values())
    
    with torch.set_grad_enabled(mode=='train'):
        with tqdm(total=total_samples, desc=mode.capitalize()) as pbar:
            while any(queues.values()):
                available = [cl for cl in queues if queues[cl]]
                selected_cl = available[0] if len(available) == 1 else random.choice(available)
                
                batch = [queues[selected_cl].pop() for _ in range(min(bsize, len(queues[selected_cl])))]
                
                TF = torch.stack([p.TF_embed for p in batch]).cuda()
                Target = torch.stack([p.Target_embed for p in batch]).cuda()
                labels = torch.tensor([p.link_label for p in batch], dtype=torch.long).cuda()
                cl_embeds = torch.stack([cell_line_embeddings[selected_cl]] * len(batch)).cuda()
  
                logits, loss, acc = classifier(TF, Target, labels, cl_embeds)
                
                if mode == 'train' and optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                all_targets.extend(labels.cpu().numpy())
                all_scores.append(logits.detach().cpu())
                total_loss.append(np.array([loss.item(), acc]) * len(batch))
                pbar.update(len(batch))
    
    avg_loss = np.sum(total_loss, axis=0) / max(1, total_samples)
    all_scores = torch.cat(all_scores).numpy() if all_scores else np.array([])
    all_targets = np.array(all_targets) if all_targets else np.array([])
    
    auc = roc_auc_score(all_targets, all_scores) if len(np.unique(all_targets)) > 1 else 0
    ap = average_precision_score(all_targets, all_scores) if len(all_targets) > 0 else 0
    
    return np.concatenate((avg_loss, [auc, ap]))