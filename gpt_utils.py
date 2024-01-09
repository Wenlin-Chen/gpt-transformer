import torch


def process_text(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_idx = {char:idx for idx, char in enumerate(chars)}
    idx_to_char = {idx:char for idx, char in enumerate(chars)}

    encode = lambda string: [char_to_idx[char] for char in string]
    decode = lambda idxs: "".join([idx_to_char[idx] for idx in idxs])

    return vocab_size, encode, decode


def get_batch(data, batch_size, block_size, device):
    idxs = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in idxs]).to(device)
    y = torch.stack([data[i+1: i+block_size+1] for i in idxs]).to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, data_splits, eval_iters, batch_size, block_size, device):
    out = {}
    model.eval()
    for split, data in data_splits.items():
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out