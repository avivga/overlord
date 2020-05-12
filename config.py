base_config = dict(
	content_dim=128,
	class_dim=256,

	content_std=0,
	content_decay=0,

	n_adain_layers=4,
	adain_dim=256,

	train=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			latent=1e-3,
			generator=1e-4,
			discriminator=1e-4,
			min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			content=0,
			adversarial=0.1
		)
	)
)
