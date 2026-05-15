import torch
import torch.nn.functional as F
def generate_mixed_label(outputs, true_label, mask, unlab=False):
    if unlab:
        true_mask = 1 - mask
        pred_mask = mask
    else:
        true_mask = mask
        pred_mask = 1 - mask
    
    outputs_soft = torch.softmax(outputs, dim=1)
    pred_label = torch.argmax(outputs_soft, dim=1)  # (B, 1, X, Y, Z)
    
    mixed_label = true_label * true_mask + pred_label * pred_mask  # (B, 1, X, Y, Z)
    
    return mixed_label



def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    
    # print(tensor.max())
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def getPrototype_o(fts, mask, region=False):
    """
    Average the features to obtain the prototype

    Args:
        fts: input features, expect shape: B x Channel x X x Y x Z
        mask: binary mask, expect shape: B x class x X x Y x Z
        region: focus region, expect shape: B x X x Y x Z
    """
    num_classes = mask.shape[1]
    batch_size = mask.shape[0]
    if torch.is_tensor(region):
        features = [[getFeatures(fts[B,...], mask[B,C,...], region[B,...]) for B in range(batch_size)] for C in range(num_classes)]
    else:
        features = [[getFeatures(fts[B,...], mask[B,C,...]) for B in range(batch_size)] for C in range(num_classes)]
    prototypes = [torch.unsqueeze(torch.sum(torch.cat(class_fts),dim=0),0) / batch_size  for class_fts in features]
    # prototypes = 
    return prototypes

def getPrototype(fts, mask, region=False):
    """
    Average the features to obtain the prototype

    Args:
        fts: input features, expect shape: B x Channel x X x Y x Z
        mask: binary mask, expect shape: B x class x X x Y x Z
        region: focus region, expect shape: B x X x Y x Z
    """
    num_classes = mask.shape[1]
    batch_size = mask.shape[0]
    if torch.is_tensor(region):
        features = [[getFeatures(fts[B,...], mask[B,C,...], region[B,...]) for B in range(batch_size)] for C in range(num_classes)]
    else:
        features = [[getFeatures(fts[B,...], mask[B,C,...]) for B in range(batch_size)] for C in range(num_classes)]
    prototypes = [torch.stack(class_fts, dim=0) for class_fts in features]

    return prototypes

def getFeatures(fts, mask, region=False):
    """
    Extract foreground and background features via masked average pooling

    Args:
        fts: input features, expect shape: C x X' x Y' x Z'
        mask: binary mask, expect shape: X x Y x Z
    """
    fts = torch.unsqueeze(fts, 0)
    if torch.is_tensor(region):
        mask = torch.unsqueeze(mask * region, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2,3,4))
    else:
        mask = torch.unsqueeze(mask, 0)
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3, 4)) \
            / (mask[None, ...].sum(dim=(2, 3, 4)) + 1e-5) # 1 x C
    return masked_fts

def calDist(fts, prototype, scaler=1.):
    """
    Calculate the distance between features and prototypes

    Args:
        fts: input features
            expect shape: N x C x X x Y x Z
        prototype: prototype of one semantic class
            expect shape: 1 x C
    """
    dist = F.cosine_similarity(fts, prototype[..., None, None, None], dim=1) * scaler
    return dist