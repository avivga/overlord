base_config = dict(
	content_depth=1,
	class_depth=256,

	content_std=0,

	train=dict(
		batch_size=64,
		n_epochs=1000,

		learning_rate=dict(
			latent=1e-3,
			generator=1e-4,
			discriminator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0,
			adversarial=0,
			gradient_penalty=0
		)
	)
)
