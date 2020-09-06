import argparse
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras import optimizers, losses, utils


def build_classifier(code_dim, n_targets, reg=0.01):
	code = Input(shape=(code_dim, ))

	x = Dense(units=256, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(code)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	x = Dense(units=256, kernel_regularizer=l2(reg), bias_regularizer=l2(reg))(x)
	x = BatchNormalization()(x)
	x = LeakyReLU()(x)

	p = Dense(units=n_targets, kernel_regularizer=l2(reg), bias_regularizer=l2(reg), activation='softmax')(x)
	model = Model(inputs=code, outputs=p)

	model.compile(
		loss=losses.categorical_crossentropy,
		optimizer=optimizers.adam(lr=1e-3),
		metrics=['accuracy']
	)

	model.summary()
	return model


def classify(args):
	data = np.load(args.data_path)
	X, y = data[args.source], data['class_ids']

	scaler = StandardScaler()
	X = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.validation_split)

	n_samples, code_dim = X.shape
	n_targets = np.unique(y).size

	classifier = build_classifier(code_dim, n_targets)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', min_delta=1e-3, factor=0.5, patience=5, verbose=1)
	early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-3, patience=10, verbose=1)

	classifier.fit(
		x=X_train, y=utils.to_categorical(y_train, n_targets),
		validation_data=(
			X_test, utils.to_categorical(y_test, n_targets)
		),
		batch_size=64, epochs=1000,
		callbacks=[reduce_lr, early_stopping],
		verbose=1
	)

	train_loss, train_accuracy = classifier.evaluate(x=X_train, y=utils.to_categorical(y_train, n_targets))
	print('train accuracy: %f' % train_accuracy)

	validation_loss, validation_accuracy = classifier.evaluate(x=X_test, y=utils.to_categorical(y_test, n_targets))
	print('validation accuracy: %f' % validation_accuracy)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-path', type=str, required=True)
	parser.add_argument('--validation-split', type=float, default=0.2)
	parser.add_argument('--source', type=str, choices=['content_codes', 'style_codes'], required=True)

	args = parser.parse_args()
	classify(args)
