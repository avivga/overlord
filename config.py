base_config = dict(
	content_dim=64,
	class_dim=512,
	style_dim=64,

	content_std=1,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	style_descriptor=dict(
		layer=7,  # 21
		dim=2*128  # 512
	),

	train=dict(
		batch_size=32,
		n_epochs=1000,

		learning_rate=dict(
			latent=1e-2,
			generator=1e-3
		),

		loss_weights=dict(
			reconstruction=1,
			content_decay=0.01
		)
	),

	gan=dict(
		batch_size=8,
		n_epochs=1000,

		learning_rate=dict(
			generator=1e-4,
			discriminator=1e-4
		),

		loss_weights=dict(
			reconstruction=1,
			latent=10,
			adversarial=1,
			gradient_penalty=1
		)
	)
)
