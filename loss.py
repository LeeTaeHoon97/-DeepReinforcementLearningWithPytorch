import torch

def softmax_cross_entropy_with_logits(y_true, y_pred):

	p = y_pred
	pi = y_true

	zero = torch.zeros(pi.shape, dtype=torch.float32)
	where = torch.equal(pi, zero)

	negatives = torch.full(pi.shape, -100.0)
	p = torch.where(where, negatives, p)

	loss = torch.nn.softmax_cross_entropy_with_logits(labels = pi, logits = p)

	return loss


