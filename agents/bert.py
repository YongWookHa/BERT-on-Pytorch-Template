import numpy as np
from tqdm import tqdm

import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from agents.base import BaseAgent
from datasets.bert import SentencePairDataset
from graphs.models.bert import BERTModel4Pretrain
from utils.optim import optim4GPU
from utils.tokenization import FullTokenizer
from utils.misc import set_seeds
''
cudnn.benchmark = True


class BERTAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        set_seeds(self.config.seed)
        self.current_epoch = 0
        self.global_step = 0
        self.best_valid_mean_iou = 0

        self.model = BERTModel4Pretrain(self.config)
        self.criterion1 = nn.CrossEntropyLoss(reduction='none')
        self.criterion2 = nn.CrossEntropyLoss()

        self.optimizer = optim4GPU(self.config, self.model)
        self.writer = SummaryWriter(log_dir=self.config.log_dir)

        tokenizer = FullTokenizer(self.config, do_lower_case=True)
        tokenizer.vocab
        train_dataset = SentencePairDataset(self.config, tokenizer, 'train')
        test_dataset = SentencePairDataset(self.config, tokenizer, 'validate')

        
        a = train_dataset.__getitem__(0)

        self.train_dataloader = DataLoader(train_dataset,
                                            batch_size = self.config.batch_size,
                                            num_workers = self.config.data_loader_workers,
                                            pin_memory = self.config.pin_memory
                                            )

        self.test_dataloader = DataLoader(test_dataset,
                                            batch_size = self.config.batch_size,
                                            num_workers = self.config.data_loader_workers,
                                            pin_memory = self.config.pin_memory)                                            

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")
        

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        # Save the state
        state = {
            'epoch': self.current_epoch,
            'iteration': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + file_name)

        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + file_name,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """
        self.model.train()
        self.load_checkpoint(self.config.checkpoint_to_load)
        if self.config.gpu_cpu == 'gpu':
            self.model = self.model.to(self.config.gpu_device)
        elif self.config.gpu_cpu == 'cpu':
            self.model.cpu()
        if self.config.data_parallel: # use Data Parallelism with Multi-GPU
            self.model = nn.DataParallel(self.model)

        self.global_step = 0
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.save_checkpoint(file_name="bert_checkpoint_epoch_{}.tar".format(epoch))

    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """

        iter_bar = tqdm(enumerate(self.train_dataloader), 
                    desc="Iter (loss=X.XXX)")

        loss_sum = 0.  # the sum of iteration losses to get average loss in every epoch
        for i, batch in iter_bar:
            if self.config.gpu_cpu == 'gpu':
                batch = [t.to(self.config.gpu_device) for t in batch]
            elif self.config.gpu_cpu == 'cpu':
                batch = [t.cpu() for t in batch]
            else:
                raise NotImplementedError

            self.optimizer.zero_grad()
            loss = self.get_loss(batch).mean()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            loss_sum += loss.item()
            iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())

            if self.global_step % self.config.save_steps == 0: # save
                self.save_checkpoint(file_name="bert_checkpoint_global_step_{}.tar".format(self.global_step))

            if self.config.total_steps and self.config.total_steps < self.global_step:
                print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.config.n_epochs, loss_sum/(i+1)))
                print('The Total Steps have been reached.')
                # save and finish when global_steps reach total_steps
                self.save_checkpoint(file_name="bert_checkpoint_global_step_{}.tar".format(self.global_step)) 
                return

        print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.config.n_epochs, loss_sum/(i+1)))
            
    
    def get_loss(self, batch): # make sure loss is tensor
        bert_input, segment_label, input_mask, bert_label, masked_pos, masked_weights, is_next = batch

        logits_lm, logits_clsf = self.model(bert_input, segment_label, input_mask, masked_pos)
        loss_lm = self.criterion1(logits_lm.transpose(1, 2), bert_label) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = self.criterion2(logits_clsf, is_next) # for sentence classification
        self.writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_total': (loss_lm + loss_clsf).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           self.global_step)
        return loss_lm + loss_clsf

    def validate(self, m):
        """
        One cycle of model validation
        :return:
        """
        self.model.eval()
        self.load(model_file, None)

        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        raise NotImplementedError