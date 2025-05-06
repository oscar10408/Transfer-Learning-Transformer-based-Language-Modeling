import torch
from torch.utils.data.dataloader import DataLoader

class Trainer:
    def __init__(self, model, 
                 dataset, 
                 learning_rate,
                 batch_size=64,
                 max_iters=50000,
                 log_interval=500):
        self.model = model
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"running on {self.device}")

        self.batch_size = batch_size
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if any(key in name for key in ['emb', 'pos', 'ln', 'bias']):
                no_decay.append(param)
            else:
                decay.append(param)
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                            {"params": no_decay, "weight_decay": 0.0}], 
                                            lr=self.learning_rate, 
                                            betas=(0.9, 0.95))
        self.log_interval = log_interval

    def run(self, type):
        assert type in ['multiplication', 'story']
        train_loader = DataLoader(  self.dataset,
                                    sampler=torch.utils.data.RandomSampler(self.dataset, 
                                                                           replacement=True, 
                                                                           ),
                                    shuffle=False,
                                    pin_memory=True,
                                    batch_size=self.batch_size,
                                )

        self.model.train()
        self.iter_num = 0
        self.train_losses = []
        data_iter = iter(train_loader)
        self.model = self.model.to(self.device)
        for _ in range(self.max_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device) # x: input, y: target
            ###########################################################################
            # TODO: Train self.model on the sampled batch in PyTorch                  #
            # Hint: 1. Zero the gradients of self.model, with set_to_none=True        #
            #       2. Calculate loss given input x and target y                      #
            #       3. Compute gradients from loss                                    #
            #       4. Clip the gradient norm with                                    #
            #           torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  #
            #       5. Perform an optimization step on self.optimizer                 #
            ###########################################################################
            self.model.zero_grad(set_to_none = True)

            output, _ = self.model(x)
            output = torch.transpose(output, 1, 2)
            loss = torch.nn.CrossEntropyLoss()(output, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()
            ###########################################################################
            #                             END OF YOUR CODE                            #
            ###########################################################################
            self.train_log(loss, type)
            self.iter_num += 1


    def train_log(self, loss, type):
        self.train_losses.append(loss.item())
        self.model.eval() #set model to eval mode
        if (self.iter_num % self.log_interval == 0) or ((self.iter_num+1) >= self.max_iters):
            print(f"Iteration {self.iter_num+1}/{self.max_iters}: training loss {loss.item():.4f}")
            if type == 'multiplication':
                Evaluator(self.dataset, self.model, 'train').eval_split(self.device)
            elif type == 'story':
                contexts = ["Once upon a time",
                            "Once there was an ",
                            "One day"]
                for i in range(3):
                    with torch.no_grad():
                        x = torch.tensor([[self.dataset.stoi[s] for s in contexts[i]]]).long()
                        y = self.model.generate(x.to(self.device), 
                                                1024, 
                                                sampling=True, 
                                                top_k=10)[0]
                        story = ''.join([self.dataset.itos[int(i)] for i in y])
                        story = story.split('<|endoftext|>')[0]
                    print(f"Story ({i+1}): \n{story}")
        self.model.train()

class Evaluator:
    def __init__(self, dataset, model, split):
        self.model = model
        self.dataset = dataset
        self.split = split

    def eval_split(self, device, print_example=0):
        self.model = self.model.to(device)
        n = self.dataset.n
        results, all_preds = [], []
        
        loader = DataLoader(self.dataset, 
                            batch_size=64, 
                            num_workers=0, 
                            shuffle=True,
                            drop_last=False)
        for x, _ in loader:
            B = x.size(0)
            ab = x[:, :n*2].to(device)
            abc = self.model.generate(ab, 2*n)
            a, b = self.dataset.str_to_digit(ab[:,:n], device), self.dataset.str_to_digit(ab[:,n:n*2], device)
            c_gt = a * b
            c_pred = self.dataset.str_to_digit(abc[:, -(2*n):].flip(1), device)
            all_correct = (c_pred == c_gt).cpu()

            for i in range(B): #batch
                results.append(int(all_correct[i]))
                all_preds.append((a[i], b[i], c_pred[i], c_gt[i]))

        print(f'{self.split} accuracy: {100*sum(results)/len(results):.2f}%')
        for (a[i], b[i], c_pred[i], c_gt[i]) in all_preds[:print_example]:
            print(f"Prediction: {a[i]} x {b[i]} = {c_pred[i]}, correct answer: {c_gt[i]}")

