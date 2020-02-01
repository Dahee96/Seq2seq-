import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
#from torchtext.data.metrics import bleu_score
import torch.nn.functional as F
import spacy
import numpy as np
import random
import math
import time


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#exp tag
#tag="nlp"
#mkdir -p exp/tag



#load model> de: German, en: English
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)
TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)


train_data, valid_data, test_data = Multi30k.splits(exts=('.de','.en'),
                                                    fields=(SRC, TRG))
print("Number of training examples: %d" %len(train_data.examples))
print("Number of validation examples: %d" %len(valid_data.examples))
print("Number of testing examples: %d" %len(test_data.examples))

print(vars(train_data.examples[0]))


SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print("Unique tokens in source (de) vocabulary: %d " %len(SRC.vocab))
print("Unique tokens in target (en) vocabulary: %d " %len(TRG.vocab))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),batch_size = BATCH_SIZE, device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim*2, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input):
        ##packed_outputs, hidden = self.rnn(enc_input)
        padded_enc_input = nn.utils.rnn.pad_packed_sequence(enc_input)
        outputs, (hidden, cell) = self.rnn(padded_enc_input)
        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch
        ##outputs, (hidden, cell) = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        return hidden, cell

    def embedding(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [src len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)
        return packed_embedded

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.rnncell=nn.LSTMCell
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded,(hidden,cell))
        # output = [1, batch size, hid dim]
        # hidden = [n layer, batch size, hid dim] #st
        # cell = [n layer, batch size, hid dim]   #ct
        prediction = self.fc_out(output.squeeze(0))
        #prediction = [batch size, output dim]
        return prediction, hidden, cell

    def embedding(self, trg):
        conv_embedded = self.dropout(self.embedding(trg))
        #trg 패딩해야하는지 모르겠네...packed_embedded = nn.utils.rnn.pack_padded_sequence(conv_embedded, trg_len)
        return conv_embedded


class CNNnet(nn.Module):
    def __init__(self, output_dim, input_dim, emb_dim, hid_dim, conv_layers, kernel_size, dropout, trg_pad_idx, device):
        RecursionError: maximum recursion depth exceeded
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.fc_out = nn.Linear(output_dim, input_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(conv_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, conv_embedded):
        trg_len =trg.shape[0]
        # conv_embedded = [trg len, batch size, emb dim]
        conv_input = self.emb2hid(conv_embedded)
        # conv_input = [trg len, batch size, hid dim]
        conv_input = conv_input.permute(1,2,0)
        #conv_input = [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0] 
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, hid_dim, self.kernel_size-1).fill_(self.trg_pad_idx).to(self.device)
            paddded_conv_input = torch.cat((padding, conv_input), dim=2)
            # padded_conv_input = [batch size, hid dim, trg len + kernel size-1]
            conved = conv(paddded_conv_input)
            # conved = [batch size, 2*hid dim, trg len]
            conved = F.glu(conved, dim=1)
            # conved = [batch size, hid dim, trg len]
            conv_input = conved
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # conved = [batch size, trg len, emb dim]
        print('conved.shape', conved.shape)
        trg_conv = self.fc_out(self.dropout(conved)) # permute해서 바꾼다음에 해야하나 끝 dimension이 바뀌는건가
        # trg_conv=[batch size, src len, emb dim] #이렇게 예상
        print('trg_conv.shape',trg_conv.shape)
        return trg_conv

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cnnnet, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.cnnnet = cnnnet
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [scr len, batch size]
        # trg = [trg len, batch size]

        batch_size = trg.shape[1]
        src_len = src.shape[0]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
       
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)


        # last hidden state of the encoder is used as the initial hidden state of the decoder
        packed_embedded = self.encoder.embedding(src, src_len)
        # econder랑 conv랑 concat
        conv_embedded = self.decoder.embedding(trg)
        trg_conv = self.cnnnet(conv_embedded)
        enc_input = torch.cat((packed_embedded, trg_conv), dim=2)
        hidden, cell = self.encoder(enc_input)

        # first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1,trg_len):

            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

#Training the Seq2Seq Model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
KERNEL_SIZE = 3
CONV_LAYERS = 2
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPPUT = 0.5
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPPUT)
net = CNNnet(OUTPUT_DIM, INPUT_DIM, ENC_EMB_DIM, HID_DIM, CONV_LAYERS, KERNEL_SIZE, ENC_DROPOUT,TRG_PAD_IDX, device)
model = Seq2Seq(enc, dec, net, device).to(device)

#initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)


# optimize
optimizer = optim.Adam(model.parameters())

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

# Training model
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        output = model(src, trg)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        # output = [(trg len -1) * batch size, output dim]
        # trg = [(trg len -1) * batch size]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# evaluate
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
          
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# epoch이 얼마나 걸렸는지 시간 측정
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1
CLIP = 1 
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    print('Epoch: %d | Time: %d 분 %d 초' %(epoch, epoch_mins, epoch_secs))
    print('\tTrain Loss: %.3f | Train PPL: %7.3f' %(train_loss, math.exp(train_loss)))
    print('\t Val. Loss: %.3f | Val. PPL: %7.3f' %(valid_loss, math.exp(valid_loss)))


model.load_state_dict(torch.load('tut1-model.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print('\tTest Loss: %3f | Test PPL: %7.3f' %(test_loss, math.exp(test_loss)))


******************************************************************************************
Traceback (most recent call last):
  File "/home/dh/PycharmProjects/seq2seq/nlp_s2s_cnnaccum.py", line 240, in <module>
    net = CNNnet(OUTPUT_DIM, INPUT_DIM, ENC_EMB_DIM, HID_DIM, CONV_LAYERS, KERNEL_SIZE, ENC_DROPOUT,TRG_PAD_IDX, device)
  File "/home/dh/PycharmProjects/seq2seq/nlp_s2s_cnnaccum.py", line 141, in __init__
    self.emb2hid = nn.Linear(emb_dim, hid_dim)
  File "/home/dh/PycharmProjects/seq2seq/venv/lib/python3.5/site-packages/torch/nn/modules/module.py", line 611, in __setattr__
    "cannot assign module before Module.__init__() call")
AttributeError: cannot assign module before Module.__init__() call
    저번코드는 encoder에서 cnn해주었고 이건 CNNnet함수를 새로 만들었는데 자꾸 이 에러가 떠요 ㅠㅠ 왜 CNNnet이라는 클래스를 못불러올까요...? encoder, decoder, cnnnet함수 순서 안꼬이는거 같은데...
    
    
    
    
    
    
