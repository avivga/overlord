base_config = dict(
	content_depth=512,
	style_dim=64,

	content_std=1,

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
			adversarial=1,
			gradient_penalty=1,
			style_reconstruction=1,
			diversity=2
		),

		n_diversity_iterations=100000
	)
)
