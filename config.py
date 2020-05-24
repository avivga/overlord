base_config = dict(
	style_dim=64,

	content_std=1,
	style_std=0,

	train=dict(
		batch_size=8,
		n_epochs=1000,

		learning_rate=dict(
			generator=1e-4,
			discriminator=1e-4,
			style_encoder=1e-4,
			mapping=1e-6
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=1e-4,
			style_decay=0,
			adversarial=1,
			gradient_penalty=1,
			style_reconstruction=1,
			diversity=2
		)
	)
)
