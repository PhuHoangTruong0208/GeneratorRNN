import torch
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
import threading
import time
import os

# nhận hổ trợ gpu hoặc cpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# lớp chuẩn hóa văn bản (ứng dụng cho cả đầu vào và đầu ra dạng text không trật tự của model)
class NormalizeSententence(nn.Module):
    def __init__(self, x_batch, y_batch):
        super().__init__()
        self.x_batch = x_batch
        self.y_batch = y_batch
        self.__sub_run__()
    
    # chạy các quá trình cần thiết khi gọi lớp mà không cần gọi hàm
    def __sub_run__(self):
        self.normailize_batch()

    # chuẩn hóa các dấu câu (chuẩn bị cho tokenizer)
    def normalizer(self, x):
        x = x.replace("?", " ? ").replace(".", " . ").replace(",", " , ").replace(":", " : ").replace("/", " / ").replace("<", " < ")
        x = x.replace(">", " > ").replace("'", " ' ").replace('"', ' " ').replace("\\", " \\ ").replace("]", " ] ").replace("[", " [ ")
        x = x.replace("{", " { ").replace("}", " } ").replace("|", " | ").replace("+", " + ").replace("=", " = ").replace("_", " _ ")
        x = x.replace("-", " - ").replace(")", " ) ").replace("(", " ( ").replace("*", " * ").replace("&", " & ").replace("^", " ^ ")
        x = x.replace("%", " % ").replace("$", " $ ").replace("#", " # ").replace("@", " @ ").replace("!", " ! ").replace("~", " ~ ")
        x = x.replace("`", " ` ").lower()
        return x

    # chuẩn hóa các lô dữ liệu cho sạch
    def normailize_batch(self):
        x_batch, y_batch = [], []
        for i in range(len(self.x_batch)):
            x_batch.append(self.normalizer(self.x_batch[i]))
            y_batch.append(self.normalizer(self.y_batch[i]))
        self.x_batch = x_batch
        self.y_batch = y_batch
    
    # xử lý văn bản đầu ra (dạng text) của model - căn chỉnh dấu câu và viết hoa chử cái hợp lý, và email..vv
    def processing_punt_output(self, x):
        sentence_punt_processed = ""
        word = x.split()
        for i in range(len(word)):
            if sentence_punt_processed in "":
                word[i] = word[i].capitalize()
            try:
                if word[i] in "." and word[i+1] in ["com", "net", "co", "jp", "vn", "us"]:
                    sentence_punt_processed = sentence_punt_processed.strip()
                    sentence_punt_processed += word[i]
                    continue
            except:
                pass
            if word[i] in ["?", ",", "!", ".", "$", "%"]:
                sentence_punt_processed = sentence_punt_processed.strip()
                sentence_punt_processed += f"{word[i]} "
            elif word[i] in ["<", ">", "+", "=", "-", "/", "&", "*"]:
                sentence_punt_processed = sentence_punt_processed.strip()
                sentence_punt_processed += f" {word[i]} "
            elif word[i] in ["[", "(", "#"]:
                sentence_punt_processed = sentence_punt_processed.strip()
                sentence_punt_processed += f" {word[i]}"
            elif word[i] in ["]", ")"]:
                sentence_punt_processed = sentence_punt_processed.strip()
                sentence_punt_processed += f"{word[i]} "
            elif word[i] in ["@", "'"]:
                sentence_punt_processed = sentence_punt_processed.strip()
                sentence_punt_processed += word[i]
            if word [i-1] in [".", "!", "?"]:
                word[i] = word[i].capitalize()
            else:
                sentence_punt_processed += word[i] + " "
        return sentence_punt_processed

# chuyển các lô dữ liệu văn bản thành tokens
class Tokenizer(nn.Module):
    def __init__(self, x_batch, y_batch, hidden_dim, max_sequence_length=100, pad="<pad>", end="<end>", start="<start>", out="<out>", limit_total=10, start_limit=0, vocab_file="vocab.txt"):
        super().__init__()
        self.x_batch = x_batch[start_limit:limit_total]
        self.y_batch = y_batch[start_limit:limit_total]
        self.x_tensor = x_batch[start_limit:limit_total]
        self.y_tensor = y_batch[start_limit:limit_total]
        self.pad = pad
        self.end = end
        self.start = start
        self.out = out
        self.max_sequence_length = max_sequence_length
        self.vocab_file = vocab_file
        self.vocab = self.__get_vocab__()
        self.vocab_size = len(self.vocab)
        self.text_to_number = {v:k for k,v in enumerate(self.vocab)}
        self.number_to_text = {k:v for k,v in enumerate(self.vocab)}
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim).to(device)
        if os.path.exists("embedding.pth") == False:
            torch.save(self.embedding.state_dict(), "embedding.pth")
        else:
            self.embedding.load_state_dict(torch.load("embedding.pth"))
        self.__run__()
    
    # lưu thứ tự từ vựng vào file txt (thứ tự từ vựng là quan trọng vì nó sẽ định hình các token cho việc lưu model khi load lại k gặp lỗi bất đồng bộ token)
    def __get_vocab__(self):
        vocab = [self.pad, self.end, self.start, self.out] + list({c for c in " ".join(self.x_batch + self.y_batch).split()})
        if os.path.exists(f"./vocab/{self.vocab_file}") == False:
            try:
                os.mkdir("./vocab")
            except:
                pass
            with open(file=f"./vocab/{self.vocab_file}", mode="a", encoding="utf-8") as file:
                for i in range(len(vocab)-2):
                    file.write(f"{vocab[i]}\n")
            return vocab
        else:
            with open(file=f"./vocab/{self.vocab_file}", mode="r", encoding="utf-8") as file:
                vocab_readed = file.read().splitlines()
            with open(file=f"./vocab/{self.vocab_file}", mode="a", encoding="utf-8") as file:
                for v in vocab:
                    if v not in vocab_readed:
                        file.write(f"{v}\n")
            with open(file=f"./vocab/{self.vocab_file}", mode="r", encoding="utf-8") as file:
                vocab_readed = file.read().splitlines()
            return vocab_readed
    
    # chạy các phương thức khi cần thiết khi gọi lớp
    def __run__(self):
        self.normalize_sentences()
        self.tokenizer()
        self.add_token()
        self.padding()
        self.normalize_2times()
        self.x_batch = torch.tensor(self.x_batch)
        self.y_batch = torch.tensor(self.y_batch)
        self.y_tensor = self.y_batch
        self.x_tensor = self.x_batch
        self.x_batch = self.embedding(self.x_batch.to(device))
        self.y_batch = self.embedding(self.y_batch.to(device))
    
    # chuẩn hóa độ dài của tất cả câu trước khi biến chúng thành ma trận
    def normalize_sentences(self):
        for i in range(len(self.x_batch)):
            if len(self.x_batch[i].split()) > self.max_sequence_length:
                self.x_batch[i] = " ".join(self.x_batch[i].split()[:self.max_sequence_length-3])
            if len(self.y_batch[i].split()) > self.max_sequence_length:
                self.y_batch[i] = " ".join(self.y_batch[i].split()[:self.max_sequence_length-3])
    
    # chuyển lô dữ liệu thành lô token
    def tokenizer(self):
        x_batch_tokenize, y_batch_tokenize = [], []
        for i in range(len(self.x_batch)):
            x_batch_tokenize.append([self.text_to_number[token] for token in self.x_batch[i].split()])
            y_batch_tokenize.append([self.text_to_number[token] for token in self.y_batch[i].split()])
        self.x_batch = x_batch_tokenize
        self.y_batch = y_batch_tokenize
    
    # thêm các token đặc biệt vào các câu đã tokenize trong batch
    def add_token(self):
        for i in range(len(self.x_batch)):
            self.x_batch[i] = self.x_batch[i] + [self.text_to_number[self.end]]
            self.y_batch[i] = [self.text_to_number[self.start]] + self.y_batch[i] + [self.text_to_number[self.end]]
    
    # thêm đệm để tất cả câu cùng cùng chiều dài (tránh gặp lỗi khi biến chúng thành tensor)
    def padding(self):
        for i in range(len(self.x_batch)):
            paddding_tensor_x = []
            for _ in range(len(self.x_batch[i]), self.max_sequence_length):
                paddding_tensor_x.append(self.text_to_number[self.pad])
            self.x_batch[i] = self.x_batch[i] + paddding_tensor_x
            paddding_tensor_y = []
            for _ in range(len(self.y_batch[i]), self.max_sequence_length):
                paddding_tensor_y.append(self.text_to_number[self.pad])
            self.y_batch[i] = self.y_batch[i] + paddding_tensor_y
    
    # chuẩn hóa độ dài của các câu đã token lần 2 để đảm bảo k có lỗi gì sẽ xảy ra khi biến chúng thành tensor
    def normalize_2times(self):
        for i in range(len(self.x_batch)):
            if len(self.x_batch[i]) > self.max_sequence_length:
                self.x_batch[i] = self.x_batch[i][:self.max_sequence_length]
            if len(self.y_batch[i]) > self.max_sequence_length:
                self.y_batch[i] = self.y_batch[i][:self.max_sequence_length]
    
    # nhận đầu vào lẽ thay vì batch để tokenize cho đầu vào thực tế
    def tokenizer_sentence(self, x):
        tensor_sentence = []
        for word in x.split():
            try: tensor_sentence.append(self.text_to_number[word])
            except: tensor_sentence.append(self.text_to_number[self.out])
        for _ in range(len(tensor_sentence), self.max_sequence_length):
            tensor_sentence.append(self.text_to_number[self.pad])
        tensor_sentence = tensor_sentence[:self.max_sequence_length]
        return torch.tensor([tensor_sentence])

# lớp chú ý (tính toán và lọc mức độ liên quan giữa đầu ra biểu diễn "encoder" và đầu ra diễn giãi "decoder")
class Attention(nn.Module):
    def __init__(self, hidden_dims, vocab_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_dims, hidden_dims*2, bias=False)
        self.W2 = nn.Linear(hidden_dims, hidden_dims*2, bias=False)
        self.V = nn.Linear(hidden_dims*2, hidden_dims, bias=False)
        self.out = nn.Linear(hidden_dims*2, vocab_size)
    
    def forward(self, q, k):
        W1 = self.W1(q).to(device)
        W2 = self.W2(k).to(device)
        W2 = W2.unsqueeze(1)
        scores = torch.tanh(W1 + W2)
        scores = self.V(scores).to(device)
        weights = torch.softmax(scores, dim=-1)
        values = weights * q
        values = torch.sum(values, dim=1)
        concat = torch.cat([q, values], dim=-1)
        out = self.out(concat).to(device)
        return out, weights

# mô hình RNN cho nhiệm vụ sinh chuổi, với tiêu chuẩn thông thường 1 encoder và 1 decoder + attention
class ModelRNN(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(device)
        self.rnn_enc_layer = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True).to(device)
        self.rnn_dec_layer = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True).to(device)
        self.attention_output = Attention(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.states = None
    
    # lớp mã hóa (biểu diễn dữ liệu)
    def rnn_enc(self, inp):
        if self.states is None or self.states[0].size(1) != inp.size(0):
            self.states = (torch.zeros(self.num_layers, inp.size(0), self.hidden_dim).to(device), torch.zeros(self.num_layers, inp.size(0), self.hidden_dim).to(device))
        rnn_out, states = self.rnn_enc_layer(inp, (self.states[0].detach(), self.states[1].detach()))
        self.states = (states[0].to(device), states[1].to(device))
        return rnn_out.to(device), states
    
    # lớp giãi mã (diễn giãi dữ liệu)
    def rnn_dec(self, inp):
        rnn_out, states = self.rnn_dec_layer(inp, (self.states[0].detach(), self.states[1].detach()))
        self.states = (states[0].to(device), states[1].to(device))
        return rnn_out.to(device), states

    def forward(self, x_inp, argmax=False):
        x_inp = self.embedding(x_inp.to(device))
        rnn_enc_out, states_enc = self.rnn_enc(x_inp)
        rnn_dec_out, states_dec = self.rnn_dec(x_inp)
        attn_op, _ = self.attention_output(rnn_dec_out, rnn_enc_out)
        out = F.log_softmax(attn_op, dim=-1)
        if argmax:
            return torch.argmax(out, dim=-1)
        return out

# lớp RNN cộng các bước xử lý tổng thể, và chứa 1 cơ chế tự động cập nhật (- tự động cập nhật theo ý tưởng của tôi là khiến mạng tạo ra vector với nhiều đa dạng câu, ngữ cảnh hơn)
# có thể bổ sung thêm những thuật toán dự đoán các từ tốt nhất trong vector ngữ cảnh này (ví dụ phổ biến : thuật toán beam search)
class SuperRNN:
    def __init__(self, x_batch, y_batch, max_sequence_length, start_limit, limit_total_sentences, hidden_dim, num_layers, dropout, lr=0.001, epochs=3, limit_break_train=2.8, training_verbose=True, batch_size=32, model_file_name="RNN.pth"):
        self.normalize_batch = NormalizeSententence(x_batch, y_batch)
        self.tokenizer = Tokenizer(self.normalize_batch.x_batch, self.normalize_batch.y_batch, hidden_dim, max_sequence_length, start_limit=start_limit, limit_total=limit_total_sentences)
        self.x_tensor = self.tokenizer.x_tensor
        self.y_tensor = self.tokenizer.y_tensor
        self.batch_size = batch_size
        self.model = ModelRNN(hidden_dim, self.tokenizer.vocab_size, num_layers, dropout).to(device)
        # kiểm tra lưu hoặc dùng mô hình nếu có, sau đó hỏi
        self.ask = ""
        if os.path.exists(model_file_name) == True:
            self.ask = input("đã có mô hình đã được lưu, dùng hay train lại? (dùng/không) : ").strip().lower()
            if self.ask in "không":
                pass
            else:
                self.model.load_state_dict(torch.load(model_file_name, map_location=device))
        else:
            self.ask = input("bạn muốn training? (muốn/không) : ").strip().lower()
        self.model_file_name = model_file_name
        self.crt = nn.CrossEntropyLoss(ignore_index=0)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.limit_break_train = limit_break_train
        self.training_verbose = training_verbose
        self.avoid_thread_err = None
        self.loss = 1
        self.__sub_run__()
    
    # lớp training và lớp training ngầm (sẽ chạy khi gọi lớp)
    def __sub_run__(self):
        if self.ask not in ["dùng", "không"]:
            self.training(batch_size=self.batch_size, epochs=self.epochs, verbose=self.training_verbose, limit_break_train=self.limit_break_train)
        background_update = threading.Thread(target=self.auto_updates_model)
        background_update.daemon = True
        background_update.start()
    
    # lớp training
    def training(self, batch_size, epochs, verbose=False, random_batch=True, limit_break_train=1, update_1time=False):
        dataset = TensorDataset(self.x_tensor, self.y_tensor)
        data_loader = DataLoader(dataset, batch_size, shuffle=random_batch)
        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(data_loader):
                self.model.train()
                self.optim.zero_grad()
                predict = self.model(x_batch)
                loss = self.crt(predict.view(-1, self.tokenizer.vocab_size).to(device), y_batch.view(-1).to(device)).to(device)
                loss.backward()
                self.optim.step()
                self.loss = loss.item()
                if verbose:
                    print(f"Batch {i} Loss {loss.item()}")
                if update_1time:
                    self.model.to(torch.device('cpu'))
                    torch.save(self.model.state_dict(), self.model_file_name)
                    self.model.to(device)
                    return 0
            if verbose:
                print(f"complete {epoch}/{epochs}")
            if self.loss < limit_break_train:
                if verbose:
                    self.model.to(torch.device('cpu'))
                    torch.save(self.model.state_dict(), self.model_file_name)
                    self.model.to(device)
                    print("mất mát đã giảm đúng như số mong muốn nên sẽ dừng traning lập tức, quá trình hoạt động thực sẽ tiếp tục được training ngầm!")
                break
            self.model.to(torch.device('cpu'))
            torch.save(self.model.state_dict(), self.model_file_name)
            self.model.to(device)
    
    # lớp tự động training, cập nhật mô hình và lưu trọng số lại vào file
    def auto_updates_model(self, updates_1time=False):
        values_moves_random = [2, 4, 5]
        while True:
            self.avoid_thread_err = "running"
            if updates_1time:
                self.training(random.choice(values_moves_random), random.choice(values_moves_random), update_1time=True)
                self.avoid_thread_err = None
                return 0
            self.training(random.choice(values_moves_random), random.choice(values_moves_random))
            print("\nsự kiện hoạt động ngầm đã diễn ra!")
            print("tiếp tục chat : ")
            self.avoid_thread_err = None
            time.sleep(random.choice([30, 60, 240, 480]))
    
    # lớp giãi mã đàu ra mô hình (dưới dạng tokens) sang text
    def decode_to_language(self, inp):
        sentences = ""
        for word_num in inp[0]:
            word = self.tokenizer.number_to_text[int(word_num)]
            if word in self.tokenizer.end:
                break
            if word in self.tokenizer.start:
                continue
            sentences += word + " "
        output = self.normalize_batch.processing_punt_output(sentences)
        return output
    
    # lớp sinh văn bản (dự đoán)
    def predicting_progress(self, inp):
        inp = self.normalize_batch.normalizer(inp)
        inp = self.tokenizer.tokenizer_sentence(inp)
        return self.model(inp, argmax=True)

x, y = [], []
with open("data_pretrain.txt", mode="r", encoding="utf-8") as file:
    data = file.read().splitlines()[:20000]
for i in range(len(data)-1):
    x.append(data[i])
    y.append(data[i+1])

model = SuperRNN(x, y, max_sequence_length=50, start_limit=0, limit_total_sentences=10000, hidden_dim=512, num_layers=2, dropout=0.01, lr=0.001, epochs=500, limit_break_train=0.5, model_file_name="states_pretrain1.pth")
while True:
    inp = input("Bạn : ")
    predict = model.predicting_progress(inp)
    output = model.decode_to_language(predict)
    print(f"Model : {output}")
