base_config = dict(
	content_dim=128,
	class_dim=256,
	style_dim=8,

	content_std=0,

	generator=dict(
		n_adain_layers=5,
		adain_dim=256
	),

	discriminator=dict(
		n_layers=4,
		filters=64
	),

	style_encoder=dict(
		n_layers=5,
		filters=16
	),

	train=dict(
		batch_size=64,
		n_epochs=1000,

		learning_rate=dict(
			latent=1e-3,
			generator=1e-4,
			discriminator=1e-4,
			min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			content=0,
			adversarial=0.1,
			style=0
		)
	)
)
