import numpy as np
import csv
import operator
import heapq
import random
import Queue

dim = 3	# dimension of embedded
internal = {}	# stores wt vector corresponding to an internal node described by its path
external = {}	# stores path of external node
alpha = 0.1 	# learning rate

def sigmoid(z):
	return 1/(1+np.exp(-z))

class HuffmanTree:

	def __init__(self, l=None, r=None, n=None, p=None):
		self.path = ''
		self.parent = p
		self.right = r
		self.left = l
		if n is not None:
			self.score = n[1]
			self.no = n[0]
		else:
			self.score = l.score + r.score
			self.no = 'nan'

	def __gt__(self,other):
		return self.score > other.score
	def __lt__(self,other):
		return self.score < other.score
	def __ge__(self,other):
		return self.score >= other.score
	def __le__(self,other):
		return self.score <= other.score
	def __str__(self):
		if self.right is None:
			return str(self.no) + ":" + str(self.score) + ":" + str(self.path)
		return ""
		#return str(self.no) + ":" + str(self.score)

	def encode(self):
		l = Queue.LifoQueue()
		l.put(self)
		global internal
		global external
		while not l.empty():
			s = l.get()		
			if s.no is 'nan':
				internal[s.path] = np.random.uniform(low = -2/(vocab_length+dim), high = 2/(dim+vocab_length), size=(dim,1))
				s.left.path = s.path + '0'
				if s.left.right is None:
					external[s.left.no] = s.left.path
				s.right.path = s.path + '1'
				if s.right.right is None:
					external[s.right.no] = s.right.path
				l.put(s.left)
				l.put(s.right)

words = "0 1 2 3 0 1 2 3 1 2 3 0 2 3 0 0 1 2 3 1 2 3 0 1 12 3 0 1 2 3 0 3 12 1 0 1 12 3 0 1 12 3 0 1 12 3 0 0".split()
#words = "".split()
vocab = list(set(words))
vocab_length = len(vocab)

index_dict = {} # stores the indices against words.
unigram = {} # stores freq of each word for huffman table
trigrams = [] # stores a list of triads of trigrams
context_count = {} # Used to calculate prob
instance_count = {} # Used to calc prob
prob_wo_given_context = {} # Prob.
weights = [] # Weights of internal nodes

for i in xrange(vocab_length):
	index_dict[vocab[i]] = i
	weights.append(np.random.uniform(low = -2/(vocab_length+dim), high = 2/(dim+vocab_length), size=(1,dim)))

for w in words:
	unigram[index_dict[w]] = unigram[index_dict[w]]+1 if index_dict[w] in unigram else 1

for i in xrange(1,len(words)-1):
	prev, curr, next = index_dict[words[i-1]], index_dict[words[i]], index_dict[words[i+1]]
	trigrams.append((prev,curr,next))
	context_count[(prev,next)] = context_count[(prev,next)]+1 if (prev,next) in context_count else 1
	instance_count[(prev,curr,next)] = instance_count[(prev,curr,next)]+1 if (prev,curr,next) in instance_count else 1

for i in xrange(1,len(words)-1):
	prev, curr, next = index_dict[words[i-1]], index_dict[words[i]], index_dict[words[i+1]]
	prob_wo_given_context[(prev,curr,next)] = float(instance_count[(prev,curr,next)])/context_count[(prev,next)]

freq = unigram.items()
#print freq
htList = []
for item in freq:
    htList.append(HuffmanTree(n=item))
heapq.heapify(htList)
i = 0
while len(htList)>1:
	ht1 = heapq.heappop(htList)
	ht2 = heapq.heappop(htList)
	heapq.heappush(htList,HuffmanTree(l=ht1,r=ht2))

h = heapq.heappop(htList);
h.encode()
#print internal
#print external



def feedforwardNetwork(trigram):
	h = np.mean(w[trigram[0]],w[trigram[2]])
	l = []
	s = external[trigram[1]]
	b = []
	result = 1
	for a in xrange(len(s),0,-1):
		l.append(s[:-a])
		b.append(s[-a])
	for i,j in zip(l,b):
		if j=='0':
			result *= sigmoid(np.dot(internal[i].transpose(),h)) 
		else:
			result *= 1-sigmoid(np.dot(internal[i].transpose(),h))
	return result
	
def backPropNetwork():
	for t in trigrams:
		h = np.add(weights[t[0]].transpose(),weights[t[2]].transpose())/2
		l = []
		b = []
		delta_w = {}
		s = external[t[1]]
		for a in xrange(len(s)-1,0,-1):
			l.append(s[:-a])
			b.append(1 if s[a]=='0' else 0)
		# samples = [random.randint(0,sam) for i in range(19)]
		delta_w = 0
		for i,j in zip(l,b):
			sig = (sigmoid(np.dot(internal[i].transpose(),h))-j)
			internal[i] -= sig*h
			delta_w += sig*internal[i]
		weights[t[0]] -= 0.5*alpha*delta_w.transpose()
		weights[t[2]] -= 0.5*alpha*delta_w.transpose()

print '\n'
print '\n'
for w in weights:
	print w
print '\n'
for i in xrange(20):
	backPropNetwork()
print '\n'
print '\n'
for w in weights:
	print w
print '\n'
print '\n'
print index_dict

l = []
for i in range(len(weights)):
	l.append([])
	for j in range(len(weights)):
		l[-1].append(np.sum((weights[i]-weights[j])**2))
for L in l:
	print L