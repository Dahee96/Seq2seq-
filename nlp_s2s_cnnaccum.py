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
            lower=True,
            include_lengths=True)
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
    (train_data, valid_data, test_data),batch_size = BATCH_SIZE, sort_within_batch = True,
     sort_key = lambda x : len(x.src), device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim*2, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input,src_len):
        
        padded_enc_input= nn.utils.rnn.pack_padded_sequence(enc_input, src_len)
        packed_outputs,(hidden,cell) = self.rnn(padded_enc_input)
        return hidden, cell

    def embedding1(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]
        return embedded

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
        # input=[batch_size]
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

    def embedding2(self, trg):
        conv_embedded = self.dropout(self.embedding(trg))
        return conv_embedded


class CNNnet(nn.Module):
    def __init__(self, output_dim, input_dim, emb_dim, hid_dim, conv_layers, kernel_size, dropout, trg_pad_idx, device):
        super().__init__()
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                              out_channels=2 * hid_dim,
                                              kernel_size=kernel_size)
                                    for _ in range(conv_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,src, conv_embedded):
        src_len = src.shape[0]
        trg_len =conv_embedded.shape[0]
        emb_dim = conv_embedded.shape[2]
        # conv_embedded = [trg len, batch size, emb dim]
        conv_input = self.emb2hid(conv_embedded)
        # conv_input = [trg len, batch size, hid dim]
        conv_input = conv_input.permute(1,2,0)
        #conv_input = [batch size, hid dim, trg len]
        batch_size = conv_input.shape[0] #다시 정의 왜해?
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
        conved=conved.permute(0,2,1)
        # conved = [batch size, emb dim, trg len]
        print('permute conved.shape', conved.shape)
        conved = conved.reshape(batch_size*emb_dim,-1)
        # conved =[batch size * emb dim, trg len]
        fc = nn.Linear(trg_len, src_len)
        trg_conv = fc(self.dropout(conved)) # (batch size * emb dim, trg len) * (trg len, src len)
        #trg_conv=[batch size * emb dim, src len]
        trg_conv=trg_conv.view(batch_size,emb_dim,src_len )
        trg_conv = trg_conv.permute(2,0,1)
        # trg_conv=[src len, batch size, emb dim]
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

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        # src = [scr len, batch size]
        # trg = [trg len, batch size]
        # src_len=[batch size] 가 되어야함 int가 아니라

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        #embedded = self.net(src,trg)
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        embedded = self.encoder.embedding1(src)
        print('encoder embedded', embedded.shape)
        #econder랑 conv랑 concat
        conv_embedded = self.decoder.embedding2(trg)
        trg_conv = self.cnnnet(src,conv_embedded)
        enc_input = torch.cat((embedded, trg_conv), dim=2)
        hidden, cell = self.encoder(enc_input,src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0,:]

        for t in range(1, trg_len):

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
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
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
        src, src_len = batch.src
        trg = batch.trg
        optimizer.zero_grad()

        output = model(src, src_len, trg)
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
            src, src_len = batch.src
            trg = batch.trg
            output = model(src, src_len, trg, 0)
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


N_EPOCHS = 50
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


from torchtext.data.metrics import bleu_score
def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        # cut off <eos> token
        pred_trg = pred_trg[:-1]
        pred_trgs.append(pred_trg)
        trgs.append([trg])
    return bleu_score(pred_trgs, trgs)

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
print('BLEU score = %.2f' %bleu_score*100 )

#bleu =0.17
