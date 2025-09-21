import torch, torch.nn as nn
from transformers import AutoModel

class BertTokenClassifier(nn.Module):
    def __init__(self, base="distilbert-base-uncased", num_labels=9):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        logits = self.classifier(seq)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return {"loss": loss, "logits": logits}
