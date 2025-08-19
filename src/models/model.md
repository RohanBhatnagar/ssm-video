VQVAE(
  18.71 M, 99.825% Params, 159.43 GMac, 99.985% MACs, 
  (pre_vq): Conv3d(15.42 k, 0.082% Params, 31.59 MMac, 0.020% MACs, 240, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (post_vq): Conv3d(15.6 k, 0.083% Params, 31.95 MMac, 0.020% MACs, 64, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (encoder): Encoder(
    9.34 M, 49.847% Params, 19.18 GMac, 12.026% MACs, 
    (convs): ModuleList(
      (0): Conv3d(6.72 k, 0.036% Params, 55.05 MMac, 0.035% MACs, 1, 240, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (1): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
    )
    (final_conv): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (res_stack): Sequential(
      6.22 M, 33.212% Params, 12.75 GMac, 7.996% MACs, 
      (0): ResidualBlock(
        3.11 M, 16.605% Params, 6.37 GMac, 3.997% MACs, 
        (bn1): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU(0, 0.000% Params, 983.04 KMac, 0.001% MACs, )
        (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (1): ResidualBlock(
        3.11 M, 16.605% Params, 6.37 GMac, 3.997% MACs, 
        (bn1): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU(0, 0.000% Params, 983.04 KMac, 0.001% MACs, )
        (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(0, 0.000% Params, 491.52 KMac, 0.000% MACs, )
    )
  )
  (decoder): Decoder(
    9.34 M, 49.813% Params, 140.19 GMac, 87.919% MACs, 
    (res_stack): Sequential(
      6.22 M, 33.212% Params, 12.75 GMac, 7.996% MACs, 
      (0): ResidualBlock(
        3.11 M, 16.605% Params, 6.37 GMac, 3.997% MACs, 
        (bn1): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU(0, 0.000% Params, 983.04 KMac, 0.001% MACs, )
        (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (1): ResidualBlock(
        3.11 M, 16.605% Params, 6.37 GMac, 3.997% MACs, 
        (bn1): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(1.56 M, 8.300% Params, 3.19 GMac, 1.998% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU(0, 0.000% Params, 983.04 KMac, 0.001% MACs, )
        (shortcut): Identity(0, 0.000% Params, 0.0 Mac, 0.000% MACs, )
      )
      (2): BatchNorm3d(480, 0.003% Params, 983.04 KMac, 0.001% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU(0, 0.000% Params, 491.52 KMac, 0.000% MACs, )
    )
    (deconvs): ModuleList(
      (0): ConvTranspose3d(1.56 M, 8.300% Params, 25.48 GMac, 15.983% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
      (1): ConvTranspose3d(1.56 M, 8.300% Params, 101.94 GMac, 63.930% MACs, 240, 240, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1))
    )
    (to_img): Conv3d(241, 0.001% Params, 15.79 MMac, 0.010% MACs, 240, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  )
  (vq): VectorQuantizer(
    0, 0.000% Params, 0.0 Mac, 0.000% MACs, 
    (embedding): Embedding(0, 0.000% Params, 0.0 Mac, 0.000% MACs, 512, 64)
  )
)
Model FLOPS: 159.45 GMac
Model Params: 18.74 M
VQVAE(
  (pre_vq): Conv3d(240, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (post_vq): Conv3d(64, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  (encoder): Encoder(
    (convs): ModuleList(
      (0): Conv3d(1, 240, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
      (1): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
    )
    (final_conv): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    (res_stack): Sequential(
      (0): ResidualBlock(
        (bn1): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU()
        (shortcut): Identity()
      )
      (1): ResidualBlock(
        (bn1): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU()
        (shortcut): Identity()
      )
      (2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU()
    )
  )
  (decoder): Decoder(
    (res_stack): Sequential(
      (0): ResidualBlock(
        (bn1): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU()
        (shortcut): Identity()
      )
      (1): ResidualBlock(
        (bn1): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv1): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (bn2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        (relu): ReLU()
        (shortcut): Identity()
      )
      (2): BatchNorm3d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (3): ReLU()
    )
    (deconvs): ModuleList(
      (0): ConvTranspose3d(240, 240, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1))
      (1): ConvTranspose3d(240, 240, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1))
    )
    (to_img): Conv3d(240, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
  )
  (vq): VectorQuantizer(
    (embedding): Embedding(512, 64)
  )
)