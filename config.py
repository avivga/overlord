base_config = dict(
	content_dim=64,
	class_dim=64,
	style_dim=256,

	content_std=0,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		img_size=128,

		batch_size=16,
		n_epochs=200,

		learning_rate=dict(
			latent=1e-2,
			encoder=1e-4,
			generator=1e-3,
			min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0.01
		)
	),

	amortize=dict(
		img_size=256,

		batch_size=4,
		n_epochs=100,
		n_epochs_warmup=10,

		learning_rate=dict(
			generator=1e-4,
			discriminator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			latent=100,
			adversarial=1
		)
	)
)
