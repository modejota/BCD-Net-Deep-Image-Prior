代码框架：
opinions：主要是一些训练需要设置的东西，如数据集文件路径参数等等
train：训练主代码
utils：一些常用的函数工具，图片读取转换等
datasets：定义数据集加载,根据不同数据集自定义
networks：定义网络结构，可以定义不同的结构，只需要保证输出形状就好了；有两个网络KNet和DNet
loss：目标变分下界，负似然 + Gauss KL + Dirichlet KL