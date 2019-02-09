import torch
from torch import nn
from torch.nn import functional as F
from .validation import validate
from tqdm import tqdm
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self, model, device="cpu", lr=1e-4):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.TripletMarginLoss(margin=1.0)

    def step(self, batch):
        self.model.train()

        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

        a_vectors = self.model(anchor)
        p_vectors = self.model(positive)
        n_vectors = self.model(negative)

        loss = self.criterion(a_vectors, p_vectors, n_vectors)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return {"loss": loss.item()}

    def validate(self):
        return {"acc": 0}

    def train(self, dloader, n_epoch=5,
              checkpoint_dir="./", checkpoint_rate=-1,
              log_dir=None, validation_rate=100, val_dataset_path=('keys', 'queries')):
        tqdm.write("Start training...")

        writer = SummaryWriter(log_dir=log_dir)
        n_iter = 0

        for epoch in range(n_epoch):
            loss = 0
            print("  |￣￣￣￣￣￣|\n  |  EPOCH: %s  |\n  |＿＿＿＿＿＿|\n(\\__/) || \n(•ㅅ•) || \n/ 　 づ" % epoch)

            bar = tqdm(dloader)
            for i, batch in enumerate(bar):
                if validation_rate > 0 and i % validation_rate == 0:
                    print("Validate..")
                    val_score = validate(self.model, self.device, val_dataset_path[0], val_dataset_path[1])
                    writer.add_scalar('val_score', val_score, n_iter)

                out = self.step(batch)
                loss += out["loss"]
                writer.add_scalar('train_loss', out["loss"], n_iter)

                bar.set_description('BATCH %i' % i)
                bar.set_postfix(loss=out["loss"])

                if checkpoint_rate > 0 and i % checkpoint_rate == 0 and i != 0:
                    print("Save checkpoint...")
                    torch.save(self.model.state_dict(),
                               checkpoint_dir + "/checkpoint_epoch_%s_batch_%s_loss_%s.pth" %
                               (epoch, i, out["loss"]))
                    print("Checkpoint saved")

                n_iter += 1

            if validation_rate > 0:
                print("Validate..")
                val_score = validate(self.model, self.device, val_dataset_path[0], val_dataset_path[1])
                writer.add_scalar('val_score', val_score, n_iter)

            print("Save epoch final checkpoint...")
            torch.save(self.model.state_dict(),
                       checkpoint_dir + "/checkpoint_epoch_%s_final.pth" % epoch)
            print("Checkpoint saved\n")

            print("LOSS:", (loss / len(dloader)))
            print("=============================\n")
        writer.close()
