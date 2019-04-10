from tqdm import tqdm
from network import LatentModel
from tensorboardX import SummaryWriter
import torchvision
import torch as t
from torch.utils.data import DataLoader
from preprocess import collate_fn
import os

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def main():
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=False,)
    epochs = 200
    model = LatentModel(128).cuda()
    model.train()
    
    optim = t.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter()
    global_step = 0
    for epoch in range(epochs):
        dloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=16)
        pbar = tqdm(dloader)
        for i, data in enumerate(pbar):
            global_step += 1
            adjust_learning_rate(optim, global_step)
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()
            
            # pass through the latent model
            y_pred, kl, loss = model(context_x, context_y, target_x, target_y)
            
            # Training step
            optim.zero_grad()
            loss.backward()
            optim.step()
                
            # Logging
            writer.add_scalars('training_loss',{
                    'loss':loss,
                    'kl':kl.mean(),

                }, global_step)
            
        # save model by each epoch    
        t.save({'model':model.state_dict(),
                                 'optimizer':optim.state_dict()},
                                os.path.join('./checkpoint','checkpoint_%d.pth.tar' % (epoch+1)))
        
        
if __name__ == '__main__':
    main()