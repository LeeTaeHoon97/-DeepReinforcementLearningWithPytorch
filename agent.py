# %matplotlib inline

from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,Dataset

import numpy as np
import random

import MCTS as mc
from game import GameState
from loss import softmax_cross_entropy_with_logits

import config
import loggers as lg
import time

import matplotlib.pyplot as plt
from IPython import display
import pylab as pl

from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self,state,vh,ph):
        self.state = state
        self.value_head = vh
        self.policy_head = ph
    def __getitem__(self, index):
        x = self.state[index]
        y = self.value_head[index]
        z = self.policy_head[index]
        return {'data': x, 'value_head': y,"policy_head": z}
    
    def __len__(self):
        return len(self.state)


class User():
	def __init__(self, name, state_size, action_size):
		self.name = name
		self.state_size = state_size
		self.action_size = action_size

	def act(self, state, tau):
		action = input('Enter your chosen action: ')
		pi = np.zeros(self.action_size)
		pi[action] = 1
		value = None
		NN_value = None
		return (action, pi, value, NN_value)



class Agent():
	def __init__(self, name, state_size, action_size, mcts_simulations, cpuct, model):
		self.name = name

		self.state_size = state_size
		self.action_size = action_size

		self.cpuct = cpuct

		self.MCTSsimulations = mcts_simulations

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = model.to(self.device)

		self.mcts = None

		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []
		self.val_overall_loss = []
		self.val_value_loss = []
		self.val_policy_loss = []

	
	def simulate(self):

		lg.logger_mcts.info('ROOT NODE...%s', self.mcts.root.state.id)
		self.mcts.root.state.render(lg.logger_mcts)
		lg.logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.state.playerTurn)

		##### MOVE THE LEAF NODE
		leaf, value, done, breadcrumbs = self.mcts.moveToLeaf()
		leaf.state.render(lg.logger_mcts)

		##### EVALUATE THE LEAF NODE
		value, breadcrumbs = self.evaluateLeaf(leaf, value, done, breadcrumbs)

		##### BACKFILL THE VALUE THROUGH THE TREE
		self.mcts.backFill(leaf, value, breadcrumbs)


	def act(self, state, tau):

		if self.mcts == None or state.id not in self.mcts.tree:
			self.buildMCTS(state)
		else:
			self.changeRootMCTS(state)

		#### run the simulation
		for sim in range(self.MCTSsimulations):
			lg.logger_mcts.info('***************************')
			lg.logger_mcts.info('****** SIMULATION %d ******', sim + 1)
			lg.logger_mcts.info('***************************')
			self.simulate()

		#### get action values
		pi, values = self.getAV(1)

		####pick the action
		action, value = self.chooseAction(pi, values, tau)

		nextState, _, _ = state.takeAction(action)

		NN_value = -self.get_preds(nextState)[0]

		lg.logger_mcts.info('ACTION VALUES...%s', pi)
		lg.logger_mcts.info('CHOSEN ACTION...%d', action)
		lg.logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)
		lg.logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value)

		return (action, pi, value, NN_value)


	def get_preds(self, state):
		#predict the leaf
		inputToModel = torch.Tensor([self.model.convertToModelInput(state)]).to(self.device)

		
		model_eval=self.model.eval()
		preds = model_eval(inputToModel)
		value_array = preds["value_head"]
		logits_array = preds["policy_head"]
		value = value_array[0]

		logits = logits_array[0]
		
		if torch.cuda.is_available():
			value=value.cpu()
			logits=logits.cpu()

		value=value.detach().numpy()
		logits=logits.detach().numpy()
		
		allowedActions = state.allowedActions

		mask = np.ones(logits.shape,dtype=bool)
		mask[allowedActions] = False
		logits[mask] = -100

		#SOFTMAX
		odds = np.exp(logits)
		probs = odds / np.sum(odds) ###put this just before the for?

		return ((value, probs, allowedActions))


	def evaluateLeaf(self, leaf, value, done, breadcrumbs):

		lg.logger_mcts.info('------EVALUATING LEAF------')

		if done == 0:
	
			value, probs, allowedActions = self.get_preds(leaf.state)
			lg.logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.state.playerTurn, value)

			probs = probs[allowedActions]

			for idx, action in enumerate(allowedActions):
				newState, _, _ = leaf.state.takeAction(action)
				if newState.id not in self.mcts.tree:
					node = mc.Node(newState)
					self.mcts.addNode(node)
					lg.logger_mcts.info('added node...%s...p = %f', node.id, probs[idx])
				else:
					node = self.mcts.tree[newState.id]
					lg.logger_mcts.info('existing node...%s...', node.id)

				newEdge = mc.Edge(leaf, node, probs[idx], action)
				leaf.edges.append((action, newEdge))
				
		else:
			lg.logger_mcts.info('GAME VALUE FOR %d: %f', leaf.playerTurn, value)

		return ((value, breadcrumbs))


		
	def getAV(self, tau):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)
		
		for action, edge in edges:
			pi[action] = pow(edge.stats['N'], 1/tau)
			values[action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def chooseAction(self, pi, values, tau):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx==1)[0][0]
		value = values[action]

		return action, value

	def replay(self, ltmemory):
		lg.logger_mcts.info('******RETRAINING MODEL******')

		#Build model
		self.model = self.model.train()
		learning_rate = config.LEARNING_RATE
		optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate,weight_decay=1e-5)	#weight_decay = l2 regularize

		vh_criterion=nn.MSELoss().to(self.device)
		ph_criderion=softmax_cross_entropy_with_logits

		for i in tqdm(range(config.TRAINING_LOOPS)):		#####config.TRAINING_LOOPS
			#minibatch는 매 반복마다 크기가 바뀔수 있다.
			minibatch = random.sample(ltmemory, min(config.BATCH_SIZE, len(ltmemory)))

			#np를 torch.Tensor로 변경
			training_states = torch.Tensor([self.model.convertToModelInput(row['state']) for row in minibatch]).to(self.device)
			training_targets = {'value_head': torch.Tensor([row['value'] for row in minibatch]).to(self.device)
								, 'policy_head': torch.Tensor([row['AV'] for row in minibatch]).to(self.device)}

			#minibatch단위로 데이터셋이 구성될때 마다 다시 데이터 로더로 batch_size만큼 불러와 학습

			#training_states 은 tensor이고 training_targets는 dict이라 커스텀 데이터셋 사용
			dataset=MyDataset(training_states,training_targets['value_head'],training_targets['policy_head'])
			ds_loader=DataLoader(dataset,batch_size=32,shuffle=True)

			fit_history = {'loss':[0.0], 'value_head_loss': [0.0], 'policy_head_loss': [0.0]}
			for batch in ds_loader:
				optimizer.zero_grad()
				hypothesis=self.model(batch['data'])
				vh_hypo=hypothesis['value_head']
				ph_hypo=hypothesis['policy_head']

				vh_cost=vh_criterion(vh_hypo,batch['value_head'])
				ph_cost=ph_criderion(ph_hypo,batch['policy_head'])

				#cost가 2개이상일경우 어떻게 처리? -> cost의 합을 backward시킴.
				(vh_cost+ph_cost).backward()
				optimizer.step()
				# print("vh+ph : ",(vh_cost+ph_cost).item())
				# print("vh : ",(vh_cost).item())
				# print("ph : ",(ph_cost).item())
				fit_history['loss'][0]+=(vh_cost+ph_cost).item()
				fit_history['value_head_loss'][0]+=vh_cost.item()
				fit_history['policy_head_loss'][0]+=ph_cost.item()

				self.train_overall_loss.append(round(fit_history['loss'][config.EPOCHS - 1],4))
				self.train_value_loss.append(round(fit_history['value_head_loss'][config.EPOCHS - 1],4))
				self.train_policy_loss.append(round(fit_history['policy_head_loss'][config.EPOCHS - 1],4))
			
			# fit = self.model.fit(training_states, training_targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
			lg.logger_mcts.info('NEW LOSS %s', fit_history)


		plt.plot(self.train_overall_loss, 'k')
		plt.plot(self.train_value_loss, 'k:')
		plt.plot(self.train_policy_loss, 'k--')

		plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')

		display.clear_output(wait=True)
		display.display(pl.gcf())
		pl.gcf().clear()
		time.sleep(1.0)

		print('\n')
		
		# self.model.printWeightAverages()

	def predict(self, inputToModel):
		self.model.eval()
		preds = self.model(inputToModel)
		return preds

	def buildMCTS(self, state):
		lg.logger_mcts.info('****** BUILDING NEW MCTS TREE FOR AGENT %s ******', self.name)
		self.root = mc.Node(state)
		self.mcts = mc.MCTS(self.root, self.cpuct)

	def changeRootMCTS(self, state):
		lg.logger_mcts.info('****** CHANGING ROOT OF MCTS TREE TO %s FOR AGENT %s ******', state.id, self.name)
		self.mcts.root = self.mcts.tree[state.id]