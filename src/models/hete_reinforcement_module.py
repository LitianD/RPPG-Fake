from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.precision_recall import Precision
from torchmetrics.classification.precision_recall import Recall
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.classification.auroc import AUROC
import os

import json

class RlUserModule(LightningModule):
    def __init__(
        self,
        num_classes:2,
        net: torch.nn.Module,
        # classifier:torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        # bert_weighted_path:str,
        # tokenizer:str,
    ):
        super().__init__()
        
        self.save_users_id_path = '/data/zlt/python_code/fake-news-baselines/RL_data/politifact/actions/'

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        #self.classifier = classifier
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes, average='macro')
        self.test_acc = Accuracy(task="binary", num_classes=self.hparams.num_classes, average='macro')
        
        self.val_pre = Precision(task="binary", num_classes=self.hparams.num_classes, average='macro')
        self.test_pre = Precision(task="binary", num_classes=self.hparams.num_classes, average='macro')
        self.val_rec = Recall(task="binary", num_classes=self.hparams.num_classes, average='macro')
        self.test_rec = Recall(task="binary", num_classes=self.hparams.num_classes, average='macro')
        self.val_f1 = F1Score(task="binary", num_classes=self.hparams.num_classes)
        self.test_f1 = F1Score(task="binary", num_classes=self.hparams.num_classes)
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, id,text,actions,split_actions):
        #y = self.net(text,image_path)
        y = self.net(id,text,actions,split_actions)
        #y = self.classifier(x1)
        return y
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def save_id(self,preds,y,user_path):
        base_path = '/data/zlt/python_code/fake-news-baselines/notebooks/deal_test_data/gos/temp_id/'
        file_name = base_path+str(self.current_epoch)+'.txt'
        if os.path.exists(file_name):
            pass
        else:
            with open(file_name, 'w') as file:
                print(f"文件不存在，已成功创建。")
        yy = torch.argmax(y, dim=1).tolist()
        preds = preds.tolist()
        for i in range(len(yy)):
            if yy[i]==1 and preds[i]==0:
                if user_path[i] != '-1':
                    temp = user_path[i].split('/')[-1][:-5]
                    with open(file_name, 'a') as file:
                        file.write(temp+'-fn\n')
            if yy[i]==0 and preds[i]==1:
                if user_path[i] != '-1':
                    temp = user_path[i].split('/')[-1][:-5]
                    with open(file_name, 'a') as file:
                        file.write(temp+'-fp\n')
        
    def actor_critic_loss(self, rewards_steps, act_probs_steps, state_values_steps, actor_loss_list, critic_loss_list):
        #rewards_steps:[(batch, 5), (batch, 15), (batch, 30);  
        # act_probs_steps:[(batch, 5), (batch, 15), (batch, 30)];    
        # state_values_steps:[(batch, 1), (batch,5,1), (batch,15,1)]
        num_steps = len(rewards_steps)
        topk=5
        for i in range(num_steps):
            shape0 = state_values_steps[i].shape[0]
            shape1 = state_values_steps[i].shape[1]
            shape2 = topk
            
            baseline =  state_values_steps[i][:,:,None]
            if i==num_steps-1:
                # terminal_reward = self.config['alpha1'] * (1 - embedding_loss.detach()) + (1-self.config['alpha1']) * (1 - reasoning_loss.detach())
                td_error = rewards_steps[i] - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
            else:
                td_error = rewards_steps[i] + 1.0 * state_values_steps[i+1] - baseline.expand(shape0, shape1, shape2).reshape(shape0, shape1*shape2)
            
            actor_loss = - torch.log(act_probs_steps[i]) * td_error.detach()
            critic_loss = td_error.pow(2)
            actor_loss_list.append(actor_loss.mean())
            critic_loss_list.append(critic_loss.mean())
            
    def model_step(self, batch: Any):
        id,text,actions,split_actions = batch
        act_probs_steps, state_values_steps, rewards_steps, anchor_graph, all_select_action_list,con_loss = self.forward(id,text,actions,split_actions)
        actor_loss_list = []
        critic_loss_list = []
        
        self.actor_critic_loss(rewards_steps, act_probs_steps, state_values_steps, actor_loss_list, critic_loss_list)
        
        actor_losses = torch.stack(actor_loss_list).sum()
        critic_losses = torch.stack(critic_loss_list).sum()
        all_loss = actor_losses + critic_losses+con_loss
    
        # if not self.training:
        #     self.save_id(preds,y,user_path)  

        return all_loss,anchor_graph,id, all_select_action_list

    def training_step(self, batch: Any, batch_idx: int):
        loss, anchor_graph,id, all_select_action_list = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, anchor_graph,id, all_select_action_list = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # self.val_pre(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.val_rec(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.val_f1(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.val_auc(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.log("val/pre", self.val_pre, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/rec", self.val_rec, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        


    def on_validation_epoch_end(self):
        pass
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        
        # print( "epoch", self.current_epoch, "val/acc", self.val_acc.compute().item(), "val/pre", self.val_pre.compute().item(),"val/rec", self.val_rec.compute().item(),"val/f1", self.val_f1.compute().item(),"val/auc", self.val_auc.compute().item(),"tp",self.val_pre.metric_state['tp'].item(),"fp",self.val_pre.metric_state['fp'].item(),"tn",self.val_pre.metric_state['tn'].item(),"fn",self.val_pre.metric_state['fn'].item())

    def test_step(self, batch: Any, batch_idx: int):
        loss, anchor_graph,id, all_select_action_list = self.model_step(batch)
        id = id.tolist()
        for i in range(len(id)):
            temp_id = id[i]
            action_list = all_select_action_list[i]
            self.save_users_id(temp_id,action_list)

        # update and log metrics
        self.test_loss(loss)
        # self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # self.test_pre(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.test_rec(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.test_f1(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.test_auc(torch.argmax(preds, dim=1), torch.argmax(targets, dim=1))
        # self.log("test/pre", self.test_pre, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/rec", self.test_rec, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)

    
    def save_users_id(self,id,action_list):
        save_json={
            'action_list':[],
        }
        # for seq, seq_len in all_seq_list:
        #     save_json['seq_len'].append(seq_len.item())
        #     save_json['seq'].append([item.item() for item in seq])
        save_json['action_list'] = action_list
        with open(self.save_users_id_path+str(id)+'.json', 'w') as json_file:
           json.dump(save_json, json_file)
           
    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = ClipModule(None, None, None)