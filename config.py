base_config = dict(
	content_dim=64,
	class_dim=512,
	style_dim=64,

	content_std=0,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	style_descriptor=dict(
		layer=30,
		dim=2*512
	),

	train=dict(
		batch_size=16,
		n_epochs=500,

		learning_rate=dict(
			latent=1e-2,
			generator=1e-3,
			min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0.001
		)
	),

	amortize=dict(
		batch_size=8,
		n_epochs=1000,

		learning_rate=dict(
			generator=1e-4,
			discriminator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			latent=10,
			adversarial=1
		)
	)
)
