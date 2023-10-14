"""
File: interactive.py
Name: Sunny
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from util import *
from submission import *
from collections import defaultdict


def main():
	weight_dic = {}
	with open('weights', 'r') as f:
		for line in f:
			line_lst = line.split()
			# new_l = line.replace('\n', '')
			# word = new_l.split()[0]
			# weights = new_l.split()[1]
			# print(weights)
			weight_dic[line_lst[0]] = float(line_lst[1])
			# weight_dic[word] = float(weights)
		# print(weight_dic)
	util.interactivePrompt(extractWordFeatures, weight_dic)


if __name__ == '__main__':
	main()
