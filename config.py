base_config = dict(
	content_depth=64,
	class_depth=512,

	content_std=0,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=32,
		n_epochs=1000,

		learning_rate=dict(
			latent=1e-3,
			generator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0
		)
	),

	gan=dict(
		batch_size=16,
		n_epochs=1000,

		learning_rate=dict(
			generator=1e-4,
			discriminator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0,
			adversarial=1,
			gradient_penalty=1
		)
	)
)
