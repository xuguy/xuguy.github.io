# LLM from scratch notes
## Working with text
- steps: splitting texts into words -> converting words into token -> turning tokens into embedding vectors
- 
## Pretraining

- `Embedding`: The concept of converting data into a vector format is often referred to as `embedding`: 利用这种思想，我们可以把不同的数据类型（文字、视频、图片）转化为向量
  - At its core, an embedding is a mapping from discrete objects, such as words, images, or even entire documents, to **points in a continuous vector space**—the primary purpose of embeddings is to convert nonnumeric data into a format that neural networks can process.
- `retrieval-augmented generation`: Retrieval augmented generation combines generation (like producing text) with retrieval (like searching an external knowledge base) to pull relevant information when generating text.
- `Word2Vec`: trained neural network architecture to generate word embeddings by predicting the context of a word given the target word or vice versa. The **main idea**
behind `Word2Vec` is that words that appear in similar contexts tend to have similar meanings.
## Deep Learning related
- thoughts, codes and notes of the book 《深度学习入门：基本框架》
\
符号定义：$w_{12}^{(1)}$ 表示第0层（输入层）到第1层（同时也是第一层隐藏层）的参数，2表示第0层的第二个神经元，1表示第1层的第一个神经元。$b_1^{(1)}$表示计算第一层隐藏层的第一个神经元用到的偏置：\[ a_1^{(1)} = w_{11}^{(1)} x_1 + w_{12}^{(1)} x_2 + b_1^{(1)} \]
一般地（写成矩阵形式）我们有：（注意W和X的顺序）
\[ A^{(1)} = X W^{(1)} + B^{(1)} \]

    其中，\( A^{(1)} \)、\( X \)、\( B^{(1)} \)、\( W^{(1)} \) 如下所示。

    \[ A^{(1)} = \begin{pmatrix} a_1^{(1)} & a_2^{(1)} & a_3^{(1)} \end{pmatrix}, \quad X = \begin{pmatrix} x_1 & x_2 \end{pmatrix}, \quad B^{(1)} = \begin{pmatrix} b_1^{(1)} \\ b_2^{(1)} \\ b_3^{(1)} \end{pmatrix} \]

    \[ W^{(1)} = \begin{pmatrix} w_{11}^{(1)} & w_{21}^{(1)} & w_{31}^{(1)} \\ w_{12}^{(1)} & w_{22}^{(1)} & w_{32}^{(1)} \end{pmatrix} \]
- **为什么一般输出层的softmax可以被省略**：神经网络只把输出值最大的神经元所对应的类别作为识别结果。并且，即便使用softmax函数，输出值最大的神经元的位置也不会变。因此，神经网络在进行分类时，输出层的softmax函数可以省略。在实际的问题中，由于指数函数的运算需要一定的计算机运算量，因此输出层的softmax函数一般会被省略。
- 在加载预训练权重识别数字的例子中(page 76)，用到了一个建议的3层神经网络，其中输入的图像数据为$X.shape=(784,)$，$784=28\times28$，0 &rarr;1 层的参数$W_1$将输入$X$映射到第一层的50个神经元，因此$W_1.shape = (784,50)$，这样$XW_1 = (1,784)\times(784,50)=(1,50)$，1&rarr;2层的参数$W_2.shape = (50,100)$，2&rarr;3：$W_3.shape=(100,10)$，第三层神经网络输出到最终的输出层的10个神经元中，最后10个神经元的值经过softmax以后输出10个概率，最大概率的神经元对应的类别即为预测类别。这里省略的偏置的讨论，每一层的偏置（注意，只有隐藏层的输入值的计算用到了偏置）向量的维度和该层神经元的个数相当，例如第一层 (0&rarr;1)的偏置$b^{(1)}$就是一个50维向量，因为第一个隐藏层有50个神经元，$XW_1=(1,50)$。上述过程可以粗略描述为
  \[\underset{(1,784)}{X} \underset{(784,50)}{W_1} \underset{(50,100)}{W_2} \underset{(100,10)}{W_3} \rightarrow \underset{(1,10)}{Y}\]
假如说我们要同时处理n张图片，只需要输入$\underset{(100,784)}{X}$即可，这就是批处理的思路。

- **为什么要引入损失函数而不直接使用识别精度（page 93）**：
  1. 用识别精度作为指标时，参数的导数在绝大多数地方都会变成0：仅仅微调参数，是无法改善识别精度的
  2. 识别精度对微小的参数变化基本上没有什么反应，即便有反应，它的值也是不连续地、突然地变化
- **为什么不适用阶跃函数作为激活函数（page 93）**：阶跃函数的导数在绝大多数地方（除了0以外的地方）均为0。也就是说，如果使用了阶跃函数，那么即便将损失函数作为指标，参数的微小变化也会被阶跃函数抹杀，导致损失函数的值不会产生任何变化。阶跃函数就像“竹筒敲石”一样，只在某个瞬间产生变化。而sigmoid函数，如图4-4所示，不仅函数的输出（竖轴的值）是连续变化的，曲线的斜率（导数）也是连续变化的。也就是说，sigmoid函数的导数在任何地方都不为0。








## Attention Mechanism
- **notations**: **attn scores** $\omega$, **attn weight** $\alpha$
- **attention weights**(page 66): typically calcualted by softmaxing the attention scores, and it determines the extent to which a context vector depends on the different parts of the input (i.e., to what extent the network focuses on different parts of the input) = **context**
- **weight parameters**(page 66): aka **weight matrices** $W_k, W_q, W_v$ for example, are the fundamental, learned coefficients that define the network’s connections, while attention weights are dynamic, context-specific values. 
- **Why scaling the attn scores?**(page 69): The reason for the scaling by the embedding dimension size is to improve the training performance by **avoiding small gradients**. For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate. 
- **nn.Module**(page 71): a fundamental building block of PyTorch models that provides necessary functionalities for model layer creation and management.
- **新老多头注意力计算方法之区别**

    那么新的multiheadattn的维度是怎么转换怎么split的呢？
    观察，一开始初始化的W_q/W_k/W_v的维度都是(d_in, d_out)，注意，这里的d_out如果转换成head，就是单头里面的head_dim（单个头的维度）。初始化以后的W_qkv接受batch(6,3)作为输入参数，然后输出queries keys values的shape都是(6,2)。接着我们把d_out（=4）分拆成了 head_dim = d_out//num_heads -> d_out = head_dim*num_heads，而num_heads=2，d_out=4，因此head_dim=2，也就是说，每个head最后生成的context_vecs的列数是2，这正好和老方法的输入维度相对应。

    \
    由此我们可以得出：老方法的输入shape为(2,6,2)，最后2个head的输出结果concat在一起，输出context_vec的shape是(2,6,4)，因为新老等价，而新方法的初始矩阵的形状是(2,6,4)，我们需要把最后一维split开，变成了(2,6,2,2)，把多出来的一维num_heads和num_tokens交换，得到(2,2,6,2)，由4变成2x2:head 1 (2,6,2), head 2 (2,6,2)，同时对这两个heads做运算，运算方法和老的计算方法一样。这里之所以要进行transpose(1,2)就是因为split的动作会多出一个维度num_heads并且这个维度出现在我们不需要的位置上，我们真正要进行矩阵乘法运算的维度和老方法一样，都是(6,2)(对应(num_tokens,head_dim)，其中head_dim又是单头里面的d_out)。第二次transpose(2,3)等价于牢房里面的.transpose(1,2)，矩阵乘法右边的那个矩阵的最后两个维度转置，这才能做乘法。

## `torch` related
1. `requires_grad = True` check the following graph and note that one of the weight param's `requires_grad` are set to false. If we call the `grad` function, `RuntimeError` will be present: `RuntimeError`: One of the differentiated Tensors does not require grad.
2. `grad(loss, b, retain_graph=True)`: By default, `PyTorch` destroys the computation graph after calculating the radients to free memory. However, since we will reuse this computation graph shortly, we set `retain_graph=True` so that it stays in memory.
3. **batch**：对batch中的每个数据分别计算gradients后取平均
4. `.zero_grad()` why?： 我们一般用batch(or mini-batch) 进行梯度更新，在每一个batch内，每个batch的数据点dou都会进行一次forward+backward，也就是说，每个数据点都会进行一次完整的反向传播，而`torch`里面的`tensor`每次backward都会把梯度累加到之前的梯度上（这样有这样的好处），因此我们需要在每次forward后backward前都要把之前的梯度清空。

## CNN

### `im2col`的计算方式示意图
**(正好和本书的展开方向相反)示意图是把输入数据中滤波器的作用区域从头开始依次纵向展开为1列（前面的通道在上面，后面的通道依次往下排），然后再拼成一行，按照书上的描述，应该是“依次纵向展开为1行”**
二者的原理一样：指定展开方向，一个通道展开完毕后下一个通道紧随其后
![](mdfig\2025-02-10-17-56-48.png)

### 包含bias的计算方法示意图 (batchsize = 1)
![](mdfig\2025-02-10-17-34-45.png)

### 前向传播shape变换：
[参考url](https://blog.csdn.net/m0_60461719/article/details/133951221)
[参考url2.good](https://blog.csdn.net/sty945/article/details/125135444)
![](mdfig\2025-02-11-18-34-19.png)

### 反向传播：

- **affine 层的反向传播**
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial Y} (dout)\cdot W^T$$

$$\frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y} (dout)$$

### Define by run
- 什么是静态计算图？
  静态计算图(define and run)的框架需要在流转数据（加减乘除等对数据的运算）之前定义计算图，在流转数据的过程中，计算图的结构不能改变，并且需要根据语言的不同，定义不同的运算语法。
- 什么是动态计算图？
  ![](mdfig\2025-02-21-22-48-45.png)


### coding 技巧
- `W, = self.params`: 看上去和`W = self.params[0]` 等价，但实际上，`W, = self.params`只有当`self.params`里面含有1个元素时不会报错，如果`self.params`里面有多个元素，就会报错。This syntax is useful when you want to enforce that an iterable contains exactly one element and avoid accidental mistakes of using an incorrect structure.
- `self.grads[0][...] = dW`
  - `self.grads` is a list that stores the gradients.
  - `self.grads[0]` is a NumPy array initialized as `np.zeros_like(W)` in `__init__`.
  - `[...]` (Ellipsis) means "assign to all elements" within the array.
  - `dW` is the newly computed gradient.
  - 见下例子：

![](mdfig\2025-02-22-23-17-06.png) 

### 解包算符 `*var`
- 这个问题来自dezero.core 中的Variable的方法reshape，为什么有一行判断 isinstance(input[0], (list, tuple)):
- 详细解释：
1. 参数结构检查
  - ***shape 会将所有位置参数打包成元组**, 当用户调用时：
    - 若以多个整数参数传递（如reshape(2, 3)），shape为(2, 3)，长度大于1，条件不满足。
    - 若以单个元组/列表传递（如reshape((2, 3))或reshape([2, 3])），shape为((2, 3),)，长度为1，且第一个元素是元组或列表，条件满足。

2. 统一参数格式
  - 当条件满足时，代码将shape替换为第一个元素（如(2, 3)）。这使得后续的dezero.functions.reshape始终接收到一个明确的形状元组，无论用户以哪种方式调用。

3. 目的: 允许用户灵活调用：
  - reshape(2, 3)（直接传多个参数）
  - reshape((2, 3))（传一个元组）
  - 两种方式最终都会提取出形状(2, 3)，确保接口的一致性和易用性。
![](mdfig\2025-02-24-16-52-53.png)

### dezero函数与数据类型规定：
![](mdfig\2025-02-24-17-14-51.png)


### CuPY adaptation的思路：
- 不是所有的环境都有`cupy`，因此load `cupy` 时需要判断是否出现`ImportError`，有的话说明`cupy`在当前环境下不可运行。
- 新建`cuda.py`，将判断环境并load相应模块的功能添加到`cuda.py`，这样以后在别的模块里就不需要复写判断逻辑。`cuda.py`中还包括了一些用来转换数据类型的方法，例如`as_numpy`或者`get_array_module`（用来判断传入的数据类型究竟是`numpy.ndarray`还是`cupy.ndarray`）
- 将`Variable`类的`init`方法里接受的数据类型由`numpy.ndarray`一个，扩充至`numpy.ndarray`+`cupy.ndarray`；后面再调用`get_array_module`方法判断这个`Variable.data`里的数据究竟是`cupy`还是`numpy`。
- `Variable`中反向传播方法的一开始需要创建一个和输入的数据具有相同形状的储存变量，这里也需要做适配。最后加上`to_cpu`和`to_gpu`方法，用来改变`self.data`的数据类型
- 储存参数的`Layer`类：`Layer`类储存的是`Parameter`类，而`Parameter`类继承了`Layer`类，因此也具有和`Layer`类一模一样的方法，不需要做改变。往`Layer`类中添加`to_cpu`和`to_gpu`，用来改变`Layer`中参数的数据类型。
- 最后，修改`DataLoader`：将`DataLoader`读到的小批次(batch)数据发送到`cupy`，需要改进`__next__`方法：将读到的`x`和`t`转换成对应的`xp.array`，这个才是读取数据的核心方法。最后添加`to_cpu`和`to_gpu`方法，这两个方法仅仅转换`self.gpu`这个tag。
- 最后的最后，逐个修改所有函数(例如`sin`)的的计算方法（大工程，未完成，进修改了Sin），需要测试。
- 总结一下，总体思路：
  - 判断当前环境类型
  - 判读数据的类型
  - 需要保存数据的类，添加能改变数据类型的方法
  - 需要计算模块的类，添加`xp.array`或者`xp.`计算方法(`xp.sin etc.`)


- `batch_size`: The number of sequences processed in parallel during training
- `block_size`: The length of the sequences being processed by the model
- `d_model`: The number of features in the model (the size of the embeddings)
- `d_k`: The number of features per attention head. 
- `num_iter`: The total number of training iterations the model will run
- `Nx`: The number of transformer blocks, or layers, in the model. 
- `eval_interval`: The interval at which the model's loss is computed and evaluated
- `lr_rate`: The learning rate for the Adam optimizer
- `device`: Automatically set to `'cuda'` if a compatible GPU is available, otherwise defaults to `'cpu'`.
- `eval_iters`: The number of iterations over which to average the evaluation loss
- `h`: The number of attention heads in the multi-head attention mechanism
- `dropout_rate`: The dropout rate used during training to prevent overfitting
