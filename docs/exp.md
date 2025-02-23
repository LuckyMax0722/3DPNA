
# version 0
实验和plus_conv以及plus_conv_1配置一样，只不过拆分出来在lightening上跑的

# version 1
用了PNA
kernel_size = [[3,3,3], [5,5,5], [5,5,5]]

# version 1
用了PNA和fused
kernel_size = [[3,3,3], [5,5,5], [7,7,7]]