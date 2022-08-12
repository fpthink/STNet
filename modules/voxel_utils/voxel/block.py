import torch
import torch.nn as nn
class GroupCompletion(nn.Module):
	def __init__(self,inplane):
		super(GroupCompletion, self).__init__()
		self.one=nn.Sequential(
							   nn.Conv1d(inplane,2*inplane,1,1),
		                       nn.ReLU(inplace=True),
		                       nn.Conv1d(2*inplane,3*2048,1,1)
		                       )
		
	def forward(self, x):
		out=self.one(x)
		return out