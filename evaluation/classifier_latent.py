import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def classify(args):
	data = np.load(args.data_path)
	X, y = data[args.source], data['class_ids']

	X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-7)

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.validation_split)

	classifier = LogisticRegression(random_state=0)
	classifier.fit(X_train, y_train)

	acc_train = classifier.score(X_train, y_train)
	acc_test = classifier.score(X_test, y_test)

	print('train = {} | test = {}'.format(acc_train, acc_test))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', type=str, required=True)
	parser.add_argument('--validation-split', type=float, default=0.2)
	parser.add_argument('--source', type=str, choices=['content_codes', 'style_codes'], required=True)

	args = parser.parse_args()
	classify(args)
