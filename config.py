base_config = dict(
	content_dim=128,
	style_dim=128,
	class_dim=16,

	content_std=0,
	style_std=0,

	train=dict(
		batch_size=64,
		n_epochs=1000,

		learning_rate=dict(
			latent=1e-3,
			generator=1e-4,
			discriminator=1e-4,
			# min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0,
			style_decay=0,
			adversarial=1,
			gradient_penalty=1,
			# style_reconstruction=0
		)
	)
)
