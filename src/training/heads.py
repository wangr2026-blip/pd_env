"""Train-only normalization; validation chooses neural checkpoints, never test labels."""
import copy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def train_head(kind, x, frame, cfg):
    train = frame.split.eq('train').to_numpy()
    val = frame.split.eq('val').to_numpy()
    y = frame.label.to_numpy()
    options = cfg['head']
    if kind == 'logistic':
        estimator = LogisticRegression(max_iter=options['max_iter'], random_state=cfg['seed'])
    elif kind == 'mlp':
        estimator = MLPClassifier(hidden_layer_sizes=(options['hidden_dim'],),
                                  learning_rate_init=options['learning_rate'],
                                  alpha=options['weight_decay'], batch_size=min(options['batch_size'], int(train.sum())),
                                  max_iter=1, random_state=cfg['seed'], shuffle=True)
    else:
        raise ValueError(kind)
    scaler = StandardScaler().fit(x[train])
    xt = scaler.transform(x)
    logs = []
    if kind == 'logistic':
        estimator.fit(xt[train], y[train])
        from sklearn.metrics import log_loss
        logs.append({'iterations': int(estimator.n_iter_.max()),
                     'train_log_loss': float(log_loss(y[train], estimator.predict_proba(xt[train]), labels=[0, 1])),
                     'validation_log_loss': float(log_loss(y[val], estimator.predict_proba(xt[val]), labels=[0, 1]))})
    else:
        from sklearn.metrics import log_loss
        best_loss = float('inf')
        for epoch in range(options['epochs']):
            estimator.partial_fit(xt[train], y[train], classes=np.array([0, 1]))
            loss = log_loss(y[val], estimator.predict_proba(xt[val]), labels=[0, 1])
            logs.append({'epoch': epoch + 1, 'validation_log_loss': float(loss), 'train_loss': float(estimator.loss_)})
            if loss < best_loss:
                best_loss = loss
                best = copy.deepcopy(estimator)
        estimator = best
    pipeline = make_pipeline(scaler, estimator)
    return pipeline, pipeline.predict_proba(x)[:, 1], logs


def attention_pool(tokens, frame, cfg):
    """Learn token attention on training recordings, choose epoch on validation loss.

    Returns reusable recording embeddings; train this auxiliary attention+linear head
    once and use identical pooled representations for both downstream heads.
    """
    import torch
    from torch import nn
    train = np.flatnonzero(frame.split.eq('train'))
    val = np.flatnonzero(frame.split.eq('val'))
    # Streaming moments over training tokens only, avoiding an all-token concatenation.
    count = sum(len(tokens[i]) for i in train)
    mean = sum(tokens[i].astype(np.float64).sum(0) for i in train) / count
    var = sum(((tokens[i].astype(np.float64) - mean) ** 2).sum(0) for i in train) / count
    scale = np.sqrt(var).clip(1e-6)
    def tensor(index):
        return torch.tensor((tokens[index] - mean) / scale, dtype=torch.float32)
    dimension = tokens[0].shape[1]
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.score = nn.Linear(dimension, 1)
            self.classifier = nn.Linear(dimension, 1)
        def pool(self, x):
            return (self.score(x).softmax(dim=0) * x).sum(0)
        def forward(self, x):
            return self.classifier(self.pool(x)).squeeze(-1)
    model = Attention()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['head']['learning_rate'], weight_decay=cfg['head']['weight_decay'])
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(cfg['seed'])
    logs = []
    best_loss = float('inf')
    for epoch in range(cfg['head']['epochs']):
        model.train()
        order = rng.permutation(train)
        for start in range(0, len(order), cfg['head']['batch_size']):
            indices = order[start:start + cfg['head']['batch_size']]
            optimizer.zero_grad()
            loss = torch.stack([criterion(model(tensor(i)), labels[i]) for i in indices]).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float(torch.stack([criterion(model(tensor(i)), labels[i]) for i in val]).mean())
        logs.append({'epoch': epoch + 1, 'validation_log_loss': loss})
        if loss < best_loss:
            best_loss = loss
            best = copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    with torch.no_grad():
        x = np.stack([model.pool(tensor(i)).numpy() for i in range(len(tokens))])
    return x, {'state_dict': best, 'mean': mean, 'scale': scale, 'dimension': dimension, 'logs': logs}
