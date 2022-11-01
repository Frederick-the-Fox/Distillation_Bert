import torch
from torch import nn
from torch import optim
from transformers import AutoTokenizer, BertModel

class WangYC_Model(nn.Module):
    def __init__(self):
        super(WangYC_Model, self).__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained("bert-base-chinese")

        self.use_bert_classify = nn.Linear(768, 40474) 
        self.class_transfer = nn.Linear(1 * args.batch_size, 2) 
        self.sig_mod = nn.Sigmoid()
        self.sm = torch.nn.Softmax(dim = -1)

    def forward(self, ):
        sentence_tokenized = self.tokenizer(batch_sentences,
                                            truncation=True,
                                            padding=True,  
                                            max_length=30,  
                                            add_special_tokens=True)  
        input_ids = torch.tensor(sentence_tokenized['input_ids']).to(device) 
        attention_mask = torch.tensor(sentence_tokenized['attention_mask']).to(device) 
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :] 
        # logits = self.use_bert_classify(bert_cls_hidden_state)
        logits = bert_cls_hidden_state
        logits_t = torch.t(logits)
        class_t = self.class_transfer(logits_t)
        class_ori = torch.t(class_t)
        class_pre = self.use_bert_classify(class_ori)
        output = self.sm(torch.t(class_pre))

        # return self.sig_mod(linear_output)
        # output = self.sig_mod(logits)
        return output