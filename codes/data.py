import numpy as np





class FakeData:

	def __init__(self,intention_prob):
		self.intention_prob = self._convert_to_prob(intention_prob)
		self.intention_num = len(intention_prob)
		self.groups = []
		self.group_num = 0
		self.sizes = [0]
		self.group_prob_by_intention = None

	def add_group(self, intention_distribution, size):
		self.groups.append(self._convert_to_prob(intention_distribution))
		self.sizes.append(size+self.sizes[-1])

	def get_ready(self):
		self.groups = np.array(self.groups)
		self.group_num = len(self.groups)
		self.group_prob_by_intention = [self._convert_to_prob(self.groups[:,i]) for i in range(self.intention_num)]

	def generate_pairs(self,num):
		output = []		
		for _ in range(num):
			# for each pair, select an intention first
			intention = np.random.choice(self.intention_num, 1, p=self.intention_prob)[0]
			# get the probablity distribution of groups based on the intnetion selected
			selected_group_prob = self.group_prob_by_intention[intention]
			pair = []
			# Select randomly 2 UNIQUE item
			while len(set(pair)) < 2:
				# pick the group from which item is selected
				selected_group = np.random.choice(self.group_num, 1, p=selected_group_prob)[0]
				# Pick the item
				items_to_select = range(self.sizes[selected_group], self.sizes[selected_group+1])
				selected_item = np.random.choice(items_to_select, 1)[0]
				pair.append(selected_item)
			output.append(tuple(set(pair)))
		return output

	# For plotting purpose
	def get_color_list(self,colors):
		assert len(colors) == self.group_num
		output = []
		for i in range(1,len(self.sizes)):
			output += colors[i-1]*(self.sizes[i]-self.sizes[i-1])
		return output

	def _convert_to_prob(self,intention_distribution):
		total = sum(intention_distribution)
		return [i/total for i in intention_distribution]




		