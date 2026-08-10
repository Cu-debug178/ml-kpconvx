"""Example ScanObjectNN overlay: KPConvD with FastAdapter and BF16 AMP."""

_base_ = '../_base_/fastadapter.py'

model = dict(kp_mode='kpconvd')

train = dict(
    amp_enabled=True,
    amp_dtype='bfloat16',
)
