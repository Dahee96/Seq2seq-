import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
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

# class CNNnet(nn.Module):
#     def __init__(self,input_dim, emb_dim, dropout):
#         super().__init__()
#         self.dec_embedding = nn.Embedding(input_dim, emb_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, 5, padding=1)
#     def forward(self,src,trg):
#         trg = trg.unsqueeze(1)
#         # trg=[trg len, 1, batch_size] #[21, 1, 128]
#         print('1:', trg.shape, trg.type)
#         trg = trg.permute(2,1,0)
#         # trg=[batch size, 1, trg len] #[128, 1, 21]
#         print('2:', trg.shape, trg.type)
#         trg = trg.float()
#         x = F.max_pool1d(F.relu(self.conv1(trg)), 2)
#         print('3:', x.shape, x.type) #[128, 32, 10?]
#         x = F.max_pool1d(F.relu(self.conv2(x)), 2)
#         print('4:', x.shape, x.type) #[128, 64, 4?]
#         x = x.view(128, -1)
#         print('5:', x.shape, x.type) # x=[128, 256?]
#
#         linear_input = x.shape[1]
#         src_len = src.shape[0]
#         fc = nn.Linear(linear_input, src_len)
#         new_trg = fc(x)
#         print('6:', new_trg.shape, new_trg.type) # x=[128, 23] #[batch size, src len]
#         new_trg = new_trg.permute(1,0) #[src len, batch size]
#         new_trg = new_trg.long()
#         new_trg = torch.LongTensor(new_trg)
#         #new_trg.type(src.type())
#         #new_trg = new_trg.long()
#         print('7:', new_trg.shape, new_trg.type)
#         dec_embedded = self.dropout(self.dec_embedding(new_trg))
#         print('8:', dec_embedded.shape)
#         embedded = torch.cat((enc_embedded, dec_embedded), dim=2)
#         return embedded

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dec_embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim*2, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 5, padding=1)
    def forward(self, src, trg):
        #src=[src len, batch size]
        enc_embedded = self.dropout(self.embedding(src))
 ###       print('0:',src.shape, enc_embedded.shape)
        #enc_embedded=[src len, batch size, emb dim]
        # trg=[trg len,batch_size] #[21, 128]

        trg = trg.unsqueeze(1)
        # trg=[trg len, 1, batch_size] #[21, 1, 128]
        ###       print('1:', trg.shape, trg.type)
        trg = trg.permute(2,1,0)
        # trg=[batch size, 1, trg len] #[128, 1, 21]
        ### print('2:', trg.shape, trg.type)
        trg = trg.float()
        x = F.max_pool1d(F.relu(self.conv1(trg)), 2)
        ###print('3:', x.shape, x.type) #[128, 32, 10?]
        x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        ###print('4:', x.shape, x.type) #[128, 64?, 4?]
        batch = src.shape[1]  # 128
        x = x.view(batch, -1)
        ###print('5:', x.shape, x.type) # x=[128, 256?] >batch, 나머지
        conv = x.permute(1,0) # [256?,128]
        conv = conv.long()
        dec_embedded = self.dropout(self.dec_embedding(conv))
        ###print('?', dec_embedded.shape) #[256,128,256]> 나머지,batch,emb_dim

        emb = dec_embedded.shape[2] #emb_dim, 256
        dec_embedded = dec_embedded.view(batch*emb, -1) # batch*emb_dim, 나머지
        ###print('><', dec_embedded.shape)
        linear_input = conv.shape[0] #256?(나머지)
        src_len = src.shape[0] #23?
        fc = nn.Linear(linear_input, src_len)
        new_trg = fc(dec_embedded) # batch*emb_dim,나머지 * 나머지,src len
        ###print('6:', new_trg.shape, new_trg.type) # x=[128*256=32768, 23] #[batch*emb, src len]
        new_trg = new_trg.permute(1,0)
        ###print('7:', new_trg.shape, new_trg.type)
        new_trg = new_trg.view(src_len,batch,emb)
        ###print(new_trg.shape,'???') #[src_len, batch, emb_dim][23,128,256]

        #dec_embedded = self.dropout(self.dec_embedding(new_trg))

        embedded = torch.cat((enc_embedded, new_trg),dim=2)
        # embedded=[src len, batch size, emb dim * 2]

        outputs, (hidden, cell) = self.rnn(embedded)
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden=[n layers* n directions, batch size, hid dim]
        #cell=[n layers * n directions, batch size, hid dim]

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.rnncell=nn.LSTMCell
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input=[batch_size]
        # hidden=[n layers * n directions, batch size, hid dim]
        # cell=[n layers * n directions, batch size, hid dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden=[n layers, batch size, hid dim]
        # context=[n layers, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded,(hidden,cell))
        # output=[seq len, batch size, hid dim * n directions]
        # hidden=[n layers * n directions, batch size, hid dim]
        # cell=[n layers * n directions, batch size, hid dim]
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layer, batch size, hid dim] #st
        # cell = [n layer, batch size, hid dim]   #ct

        prediction = self.fc_out(output.squeeze(0))
        #prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

       # self.net = net
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [scr len, batch size]
        # trg = [trg len, batch size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #embedded = self.net(src,trg)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src,trg)

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
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPPUT = 0.5

#net = CNNnet(INPUT_DIM, ENC_EMB_DIM, ENC_DROPOUT)
enc = Encoder(INPUT_DIM, OUTPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPPUT)
model = Seq2Seq(enc, dec, device).to(device)

#initialize weights
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
model.apply(init_weights)

#파라메터 수 계산
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print('The model has %s trainable parameters' %count_parameters(model)) #13899013

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
            # trg=[trg len, batch size]
            # output=[trg len, batch size, output dim]

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            # trg=[(trg len - 1) * batch size]
            # ouput=[(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# epoch이 얼마나 걸렸는지 시간 측정
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1 #??
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





