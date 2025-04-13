#!/usr/bin/python3


# import the necessary packages
import os, sys, re, io, locale, json, pickle
import math
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

DEBUGGING=False

logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input-dir", type=str,             required=True,   help="Input dir containing pre-hashed images")
	parser.add_argument("-o", "--output",    type=str,             required=True,   help="Output dir")
	parser.add_argument("-n", "--n-threads", type=int, default=0,  required=False,  help="Number of threads (0 -> disable threading)")

	parser.add_argument("-d", "--debug", action='store_true',      required=False,  help="Enable debugging")

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

		#self.average_col = self.average.hash.reshape(-1, 1)
		#self.perceptual_col = self.perceptual.hash.reshape(-1, 1)

		id, type, n = path.name.split('-')

		self.id = id
		self.type = type
		self.n = n


def run_fastcomics(hashes, hash_length):
	start_time = time.time()

	avg_hashes = [h.average.hash.reshape(-1, 1).astype(int) for h in hashes]
	avg_mat = np.concatenate(avg_hashes, axis=1)
	avg_conf_mat = np.dot(avg_mat.T, avg_mat) + np.dot(1 - avg_mat.T, 1 - avg_mat)
	avg_conf_mat_norm = avg_conf_mat / hash_length

	end_time = time.time()
	duration = end_time - start_time

	logging.info(f"FastCOMICs:     {duration: >3.3f}s")

	return avg_conf_mat_norm


def run_fastcomics_gpu(hashes, hash_length):
	start_time = time.time()

	avg_hashes = [cp.array(h.average.hash).reshape(-1, 1).astype(int) for h in hashes]
	avg_mat = cp.concatenate(avg_hashes, axis=1)
	avg_conf_mat = cp.dot(avg_mat.T, avg_mat) + cp.dot(1 - avg_mat.T, 1 - avg_mat)
	avg_conf_mat_norm = avg_conf_mat / hash_length

	end_time = time.time()
	duration = end_time - start_time

	logging.info(f"FastCOMICs GPU: {duration: >3.3f}s")

	return avg_conf_mat_norm


def run_conventional(hashes, hash_length):
	start_time = time.time()

	avg_conf_mat_norm = []
	for i in range(len(hashes)):
		temp = []
		for j in range(i, len(hashes)):
			hash_diff = hashes[i].average - hashes[j].average
			hash_diff = hash_diff / hash_length
			temp.append(1 - hash_diff)

		avg_conf_mat_norm.append(temp)

	end_time = time.time()
	duration = end_time - start_time

	logging.info(f"Conventional:   {duration: >3.3f}s")

	return avg_conf_mat_norm


def process_in_parallel(i, a, hashes, hash_length):
	results = []

	for j in range(len(hashes)):
		hash_diff = a.average - hashes[j].average
		hash_diff = hash_diff / hash_length
		hash_diff = 1 - hash_diff
		results.append(hash_diff)

	print(f"done {i}")
	return i, results

def run_parallel(hashes, hash_length, n):
	start_time = time.time()

	pool = multiprocessing.Pool(n)

	avg_conf_mat_norm = []

	results = []
	for i in range(len(hashes)):
		try:
			result = pool.apply_async(process_in_parallel, (i, hashes[i], hashes[i:len(hashes)], hash_length))
			results.append(result)

		except Exception as e:
			logging.error(f"Error in hash: {i}")
			traceback.print_exception(*sys.exc_info())

			if DEBUGGING:
				import pdb; pdb.set_trace()

		avg_conf_mat_norm.append([None] * len(hashes))


	logging.debug("Waiting for processes to finish")
	wait = True
	while wait:
		wait = False
		for result in results:
			if not result.ready():
				wait = True
			else:
				i, diffs = result.get()

				avg_conf_mat_norm[i] = diffs

				results.remove(result)
				del result

		time.sleep(1)

	results = []

	end_time = time.time()
	duration = end_time - start_time

	logging.info(f"Parallel      : {duration: >3.3f}s")

	return avg_conf_mat_norm


if __name__ == "__main__":
	args = parse_args()

	hashes = []
	for file_name in os.listdir(args['input_dir']):
		hashes.append(Hashes(Path(args['input_dir']) / Path(file_name)))

	hash_length = len(hashes[0].average)

	# Do processing
	logging.info("Starting processing")

	conventional_mat = run_conventional(hashes, hash_length)

	#parallel_mat = run_parallel(hashes, hash_length, args['n_threads'])

	fastcomics_mat = run_fastcomics(hashes, hash_length)

	fastcomics_gpu_mat = run_fastcomics_gpu(hashes, hash_length)

	logging.info("Done.")
