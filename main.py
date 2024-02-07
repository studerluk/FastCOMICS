#!/usr/bin/python3


# import the necessary packages
import os, sys, re, io, locale, json, pickle
import traceback
import argparse
import logging

from pathlib import Path

import multiprocessing

from datetime import datetime
import time

import operator
import itertools
import functools
import copy

import collections
import string

from tabulate import tabulate

import numpy as np
import cupy as cp

import imagehash


DIFF_THRESHOLD = 0.15


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-dir",   required=True,  help="Input dir containing pre-hashed images")
	parser.add_argument("-o", "--output",      required=True,  help="Output dir")

	args = vars(parser.parse_args())

	return args


class Hashes():
	def __init__(self, path: Path):
		with open(path, 'rb') as file:
			data = pickle.load(file)

		self.average = data['average']
		self.perceptual = data['perceptual']
		self.difference = data['difference']
		self.wavelet = data['wavelet']
		self.color = data['color']
		self.crop_resistant = data['crop-resistant']

		self.average_col = self.average.hash.reshape(-1, 1)
		self.perceptual_col = self.perceptual.hash.reshape(-1, 1)

		id, type, n = path.name.split('-')

		self.id = id
		self.type = type
		self.n = n


if __name__ == "__main__":
	args = parse_args()

	hashes = []
	for file_name in os.listdir(args['input_dir']):
		hashes.append(Hashes(Path(args['input_dir']) / Path(file_name)))

	hash_length = len(hashes[0].average)

	start_time = time.time()
	avg_hashes = [h.average_col for h in hashes]
	avg_mat = np.concatenate(avg_hashes, axis=1)
	avg_conf_mat = np.dot(avg_mat.T, avg_mat) + np.dot(1 - avg_mat.T, 1 - avg_mat)
	avg_conf_mat_norm = avg_conf_mat / hash_length
	end_time = time.time()
	duration = end_time - start_time

	print(f"Matrix: {duration}s")

	start_time = time.time()
	avg_conf_mat_norm = []
	for i in range(len(hashes)):
		temp = []
		for j in range(i+1, len(hashes)):
			hash_diff = hashes[i].average - hashes[j].average
			hash_diff = hash_diff / hash_length
			temp.append(hash_diff)

		avg_conf_mat_norm.append(temp)

	end_time = time.time()
	duration = end_time - start_time
	print(f"Conventional: {duration}s")

	import pdb; pdb.set_trace()
