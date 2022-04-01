import torch

def softmax_cross_entropy_with_logits(y_true, y_pred):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	p = y_pred
	pi = y_true

	zero = torch.zeros(pi.shape, dtype=torch.float32)
	where = torch.equal(pi, zero)

	negatives = torch.full(pi.shape, -100.0)
	p = torch.where(where, negatives, p)

	loss = torch.nn.CrossEntropyLoss().to(device)
	"""
	loss = torch.nn.CrossEntropyLoss(labels = pi, logits = p)
	logits = last_layer,
	labels = target_output
	"""
	return loss(p,pi)


