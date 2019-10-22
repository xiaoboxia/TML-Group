import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np



# Loss functions

def loss_coteaching(y_1, y_2, t, forget_rate, ind):

    loss_1 = F.cross_entropy(y_1, t.squeeze(), reduce = False)

    ind_1_sorted = np.argsort(loss_1.data).cuda()

    loss_1_sorted = loss_1[ind_1_sorted]



    loss_2 = F.cross_entropy(y_2, t.squeeze(), reduce = False)

    ind_2_sorted = np.argsort(loss_2.data).cuda()

    loss_2_sorted = loss_2[ind_2_sorted]



    remember_rate = 1 - forget_rate

    num_remember = int(remember_rate * len(loss_1_sorted))



#    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
#
#    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)



    ind_1_update=ind_1_sorted[:num_remember]

    ind_2_update=ind_2_sorted[:num_remember]

    # exchange

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update].squeeze())

    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update].squeeze())



    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember


def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, noise_or_not, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
     
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id] 
        
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree, noise_or_not)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
 
        pure_ratio_1 = np.sum(noise_or_not[ind])/ind.shape[0]
        pure_ratio_2 = np.sum(noise_or_not[ind])/ind.shape[0]
    return loss_1, loss_2, pure_ratio_1, pure_ratio_2  

def loss_forget(logits, labels, forget_rate, ind):
    loss = F.cross_entropy(logits, labels, reduce=False)
#    loss = loss.cpu()
    ind_sorted = np.argsort(loss.detach()).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))
    ind_update=ind_sorted[:num_remember]
#    pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

    loss_small = loss_sorted[:num_remember]

    return torch.sum(loss_small)/num_remember,ind_update

def loss_decoupling(logits1, logits2, labels, step):
    _, pred1 = torch.max(logits1.detach(), 1)
    _, pred2 = torch.max(logits2.detach(), 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id=[]
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    int_logical_disagree_id = logical_disagree_id.astype(np.int64)
    nonzeros = np.nonzero(int_logical_disagree_id)
    nonzero_int_logical_disagree_id = int_logical_disagree_id[nonzeros]
    
    _update_step = np.logical_or(nonzero_int_logical_disagree_id,step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step).cuda())
    if len(disagree_id) > 0:
        update_logits1 = logits1[disagree_id]
        update_logits2 = logits2[disagree_id]
        update_labels = labels[disagree_id]
    else:
        update_logits1 = logits1
        update_logits2 = logits2
        update_labels = labels

    cross_entropy_1 = F.cross_entropy(update_logits1, update_labels)
    cross_entropy_2 = F.cross_entropy(update_logits2, update_labels)

    loss_1 = torch.sum(update_step*cross_entropy_1)/update_labels.size()[0]
    loss_2 = torch.sum(update_step*cross_entropy_2)/update_labels.size()[0]

    return loss_1, loss_2  

class reweight_loss(nn.Module):
    def __init__(self):
        super(reweight_loss, self).__init__()
        
    def forward(self, out, T, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]] 
            out_T = torch.matmul(T.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]] 
            beta = pro1 / pro2
            beta = Variable(beta, requires_grad=True)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)


class reweighting_revision_loss(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss, self).__init__()
        
    def forward(self, out, T, correction, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]
            T = T + correction
            T_result = T
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]    
            beta = (pro1 / pro2)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)

class MCL(nn.Module):
    # Meta Classification Likelihood (MCL)

    eps = 1e-8 # Avoid calculating log(0). Use the small value of float16.

    def __init__(self):
        super(MCL, self).__init__()
        return

    def forward(self, prob1, prob2, s_label):
        P = prob1.mul(prob2)
        P = P.sum(1)
        P.mul_(s_label).add_(s_label.eq(-1).type_as(P))
        #Weight_P = P.mul_(key[:,0])
        negLog_P = -P.add_(MCL.eps).log_()
        return negLog_P.mean()

