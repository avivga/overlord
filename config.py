base_config = dict(
	content_dim=128,
	class_dim=256,

	content_std=1,
	content_decay=1e-4,

	n_adain_layers=4,
	adain_dim=256,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			network=3e-4,
			latent=3e-3,
			min=1e-5
		)
	)
)
