
<center><h1><span style="color: #f2cf4a; font-size: 1.2em; line-height:40px">CS565600 Deep Learning<br/>DataLab Cup 4: Reverse Image Caption</span></h1></center>
<center><h3>Team22: SkyNet Zero&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h3></center>

<a id='Top'></a>
### Table of Contents

* [Problem Description](#Problem-Description)
* [Source Code Description](#Source-Code-Description)
* [Preprocessing](#Preprocessing)
    * [Image](#Image)
    * [Skip-Thought Vectors](#Skip-Thought-Vectors)
        * [Introduction](#Introduction)
        * [Text to Skip-Thought Conversion](#Text-to-Skip-Thought-Conversion)
    * [seq2seq Embedding](#seq2seq-Embedding)
* [Network Architecture](#Network-Architecture)
    * [Conditioning Augmentation](#Conditioning-Augmentation)
    * [Stage I Generator](#Stage-I-Generator)
        * [Simple Version](#Stage-I-Generator-Simple)
        * [Complex Version](#Stage-I-Generator-Complex)
    * [Stage I Discriminator](#Stage-I-Discriminator)
        * [Simple Version](#Stage-I-Discriminator-Simple)
        * [Complex Version](#Stage-I-Discriminator-Complex)
    * [Stage II Generator](#Stage-II-Generator)
    * [Stage II Discriminator](#Stage-II-Discriminator)
    * [Loss Function](#Loss-Function)
* [Experiments](#Experiments)
    * [Setup](#Setup)
    * [Training Details](#Training-Details)
    * [Evaluation](#Evaluation)
* [Result](#Result)
    * [Complex Model](#Complex-Model)
    * [Simple Model](#Simple-Model)
    * [Testing Result](#Testing-Result)
        * [Top 3 Images](#Top-3)
        * [Bottom 3 Images](#Bottom-3)
        * [Stage I v.s. Stage II](#StageI-vs-StageII)
        * [Cherry-Picked Images](#Cherry-Picking)
* [Conclusion](#Conclusion)

<a id='Problem-Description'></a>
### Problem Description
Given a description of flowers. Our goal is to generate suitable 64×64 images according to the specified colors and shapes. We trained [StackGAN](https://github.com/hanzhanggit/StackGAN) on Oxford-102 Flower Dataset, which contains [102 categories](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html) of flowers. Each flower comes along with ten captions describing its colors, shapes, size, texture, and positions of floral organs.

<a id='Source-Code-Description'></a>
### Source Code Description
* DL_comp4_22_report.md: My report
* DL_comp4_22_report.html: Report in html
* Main.ipynb: Training data retrieval, data preprocessing, Training, Testing, etc
* seq2seq.ipynb: Converting captions to sentence embedding with seq2seq
* Preprocessing.ipynb: Generating training and Testing captions in desirable format
* train_captions.txt: Generated training captions
* test_captions.txt: Generated testing captions
* images/: Images used in this report
* images/: Equations used in this report

<a id='Preprocessing'></a>
### Preprocessing

<a id='Image'></a>
#### Image
The raw image sizes vary. Simply resizing the images to squares may alter the original aspect ratio. Fortunately, the flower itself usually locates in the center of the image, so we crop the longer side of the original image. Then do the reshaping. Take the original paper as reference, we modified our preprocessing flow as follows:

1.	Central cropping to a square with side length equals to the shorter side of the original image by **tf.image.resize_image_with_crop_or_pad**
2.	Scaling each sides to 76/64 times as big as the target image’s length of sides
3.	Random flipping horizontally by **tf.image.random_flip_left_right**
4.	Random cropping to target image size by **tf.random_crop**
5.	Normalizing each pixel to \[-1, 1\]

<a id='Skip-Thought-Vectors'></a>
#### Skip-Thought Vectors

<a id='Introduction'></a>
**Introduction**  
[Skip-Thought](https://arxiv.org/pdf/1506.06726.pdf) is an unsupervised learning model of sentence embedding. It abstracts the skip-gram model of word2vec to the sentence level, i.e., it predicts the context from the current sentence. The model of Skip-Thought consists of a GRU encoder and 2 GRU decoders. Feeding a sentence into the encoder, the decoders try to reconstruct the previous sentence and the next sentence.  
The authors proposed Vocabulary Expansion to expand the encoder’s vocabulary to words it has not seen during training. Let <img alt="$V_{w2v}$" src="svgs/03fc29021278ecf0afbe05434a089a92.png?invert_in_darkmode" align="middle" width="32.827245000000005pt" height="22.381919999999983pt"/> be the word embedding space from some larger pretrained model like word2vec. Let <img alt="$V_{rnn}$" src="svgs/6f1e049413e91c535425c03a3cabecdb.png?invert_in_darkmode" align="middle" width="32.17929pt" height="22.381919999999983pt"/> be the RNN word embedding space. Vocabulary Expansion tries to construct a matrix **W**, such that <img alt="$v’=Wv\;\mbox{for}\;v’\in V_{rnn}\;\mbox{and}\;v\in V_{w2v}$" src="svgs/bda47e77ceb6de14a1bfc93b9b78fbef.png?invert_in_darkmode" align="middle" width="244.17409500000002pt" height="22.745910000000016pt"/>. Thus, any word in <img alt="$V_{w2v}$" src="svgs/03fc29021278ecf0afbe05434a089a92.png?invert_in_darkmode" align="middle" width="32.827245000000005pt" height="22.381919999999983pt"/> can now be projected into <img alt="$V_{rnn}$" src="svgs/6f1e049413e91c535425c03a3cabecdb.png?invert_in_darkmode" align="middle" width="32.17929pt" height="22.381919999999983pt"/> for encoding sentences.

<a id='Text-to-Skip-Thought-Conversion'></a>
**Text to Skip-Thought Conversion**  
We use the code from [here](https://github.com/paarthneekhara/text-to-image) to generate Skip-Thought vectors. First, download the model pretrained on BookCorpus from [here](https://github.com/ryankiros/skip-thoughts#getting-started). Then, use generate_thought_vectors.py to convert train_captions.txt and test_captions.txt into h5py format. For each caption, a 2400 dimensional vector is generated from the unidirectional encoder and another 2400 dimensional vector from the bidirectional encoder. According to the authors of Skip-Thought, the concatenation of the two vectors usually yields better results on various NLP tasks.

<a id='seq2seq-Embedding'></a>
#### seq2seq Embedding
We train a LSTM Sequence-to-Sequence Model to enhance our sentence embeddings. The original sentence is fed into the encoder to get a encoded vector (the last hidden state). We want the decoder to reconstruct the exact sentence from this vector. Note that, attention mechanism is not applicable here, since it peeks the input instead of learning to reconstruct it. After convergence, we try the following methods to get the sentence embeddings:
1. Use **get_tensor_by_name** to retrieve the last hidden state of the encoder of embedding_rnn_seq2seq:

       encoder_output = tf.get_default_graph().get_tensor_by_name("seq2seq_rnn
           /embedding_rnn_seq2seq/embedding_rnn_decoder/rnn_decoder/rnn_decoder
           /output_projection_wrapper/output_projection_wrapper/lstm_cell/concat:0")
    
2. Extract the embedding matrix **embedding_wrapper/embedding:0** from the seq2seq model. The sentence embeddings can be generated by performing inner product on this matrix and the Bag-of-words representations of the original sentences.

       for var in tf.trainable_variables():
           if var.name == 'seq2seq_rnn/embedding_rnn_seq2seq/rnn/embedding_wrapper/embedding:0':
               embedding_op = var

[Back to Top](#Top)

<a id='Network-Architecture'></a>
### Network Architecture
The architecture of StackGAN is very similar to that of [GAN-INT-CLS](https://github.com/reedscot/icml2016) with some minor differences. 

<img src="images/StackGAN.jpg"> 

<a id='Conditioning-Augmentation'></a>
### Conditioning Augmentation
The most significant difference between StackGAN and GAN-INT-CLS is the Conditioning Augmenetation process before feeding text embeddings into the generator. This innovation comes from the idea of variational autoencoder (VAE), where the latent space is mixed with random pertubations. These pertubations can be generated by the learned mean vector <img alt="$\mu$" src="svgs/07617f9d8fe48b4a7b3f523d6730eef0.png?invert_in_darkmode" align="middle" width="9.867990000000004pt" height="14.102549999999994pt"/> and the learned variance vector <img alt="$\Sigma$" src="svgs/813cd865c037c89fcdc609b25c465a05.png?invert_in_darkmode" align="middle" width="11.827860000000003pt" height="22.381919999999983pt"/>. By diversifying a text embedding, the model becomes more robust.
For reversed image caption, let the text embedding be <img alt="$\varphi_t$" src="svgs/9b0e1ecc9806900220f713c919089e97.png?invert_in_darkmode" align="middle" width="15.660480000000003pt" height="14.102549999999994pt"/>. We randomly sample latent variables from the distribution <img alt="$\mathcal{N}(\mu(\varphi_t),\Sigma(\varphi_t))$" src="svgs/8548e0ee0807f1f43d5a6473bebe5bf6.png?invert_in_darkmode" align="middle" width="116.05902pt" height="24.56552999999997pt"/> as the representations of the original sentence. Equivalently, we get a bunch of training data points for each sentence. To avoid small learned variances, and make the distribution closer to the normal distribution, the KL divergence between the distributions should be minimized. According to [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), an extra term should be added to the loss function fo the generator:
<p align="center"><img alt="$$&#10;D_{KL}(\mathcal{N}(\mu(\varphi_t),\Sigma(\varphi_t))\|\mathcal{N}(0,I))=\frac{1}{2}\sum_{j=1}^{J}(-1-\log(\sigma_j^2)+\mu_j^2+\sigma_j^2)&#10;$$" src="svgs/1d2549d781d61219383b40b3c4ba3f1c.png?invert_in_darkmode" align="middle" width="464.96834999999993pt" height="50.005725pt"/></p>

<a id='Stage-I-Generator'></a>
### Stage I Generator

<a id='Stage-I-Generator-Simple'></a>
**Simple Version:**  
Stage I Generator first concatenates the augmented caption and noise vectors. Then, it performs upsampling through a fully connected layer followed by 4 deconvolutional layers, where the output feature map is 2× higher, 2× wider and 0.5× deeper than the previous layer. Note that the deconvolutional layers should be implemented by **tf.nn.conv2d_transpose** instead of **tf.layers.conv2d_transpose**; otherwise we can't specify the desired output shape.

<p align="center"><img src="images/GeneratorI_Simple.png"></p>

<a id='Stage-I-Generator-Complex'></a>
**Complex Version:**  
The deconvolutional layers in the simple version are replaced with **tf.image.resize_nearest_neighbor**. Moreover, few strided-1 convolutional layers with residual connection are added.

<p align="center"><img src="images/GeneratorI_Complex.png"></p>

<a id='Stage-I-Discriminator'></a>
### Stage I Discriminator

<a id='Stage-I-Discriminator-Simple'></a>
**Simple Version:**  
It performs downsampling through 4 strided-2 convolutional layers, where the output feature map is 0.5× higher, 0.5× wider and 2× deeper than the previous layer. The replicated text embeddings are stacked along the channel dimension before being concatenated with the final output of the downsampling layers. Lastly, it compute the logits by a window-size convolutional layer (equvalent to a fully connected layer).

<p align="center"><img src="images/DiscriminatorI_Simple.png"></p>

<a id='Stage-I-Discriminator-Complex'></a>
**Complex Version:**  
Compared to the simple version, a residual connection and 3 strided-1 convolutional layers are added.

<p align="center"><img src="images/DiscriminatorI_Complex.png"></p>

<a id='Stage-II-Generator'></a>
### Stage II Generator
It take the 64×64 images generated in Stage I as input. A structure the same as Stage I discriminator (simple) is used for downsampling; however, all leaky ReLUs are replaced by ReLUs. The role of the following 4 residual connections is to preserve the rough characteristics that was already learned in Stage I. Lastly, the upsampling is performed by **tf.image.resize_nearest_neighbor** and strided-1 convolutional Layers. The final images are 256×256.

<p align="center"><img src="images/GeneratorII.png"></p>

<a id='Stage-II-Discriminator'></a>
### Stage II Discriminator
Compared to the simple version, more strided-2 convolutional layers are added to shrink images to the size the same as Stage I. Moreover, several strided-1 convolutional layers and a residual connection is inserted before the logits layer.  

<p align="center"><img src="images/DiscriminatorII.png"></p>

<a id='Loss-Function'></a>
### Loss Function
Different from the vanilla generator loss, an additional KL divergence loss is needed as afrontmentioned. As for the discriminator loss, apart from the real loss (real image with real text) and fake loss (fake image with real text), the wrong loss (real image with fake text) is needed; otherwise, it's possible that the generator generates images well disregarding of the provided captions. In practice, we shuffle the mapping relationships between images and captions instead of generating fake captions.  
[Back to Top](#Top)

<a id='Experiments'></a>
### Experiment

<a id='Setup'></a>
#### Setup

| Hyperparameter    | Value           | Hyperparameter     | Value            |
| ----------------- | --------------- | ------------------ | ---------------- |
| FC Initializer    | N(0, 0.02)      | Conv Initializer   | clip(N(0, 0.02)) |
| Gamma Initializer | N(1, 0.02)      | Deconv Initializer | N(0, 0.02)       |
| Z Dimension       | 100             | KL Loss Weight     | 2                |
| Learning Rate     | 0.002           | Optimizer          | Adam             |
| Decay Step        | 50              | <img alt="$\beta$" src="svgs/8217ed3c32a785f0b5aad4055f432ad8.png?invert_in_darkmode" align="middle" width="10.127700000000003pt" height="22.745910000000016pt"/> from Adam  | 0.5              |

<a id='Training-Details'></a>
#### Training Details
1.	Following the practice of GAN-INT, each sample of text embedding is the average vector of 4 random captions out of the 10 captions. Therefore, the total samples per image is <img alt="$C_4^{10}$" src="svgs/607f6918b970d4240905190f3b810bcc.png?invert_in_darkmode" align="middle" width="25.936844999999998pt" height="26.70657pt"/>.
2.	Using of **tf.layers.batch_normalization** automatically add two varibles, moving mean and moving variance to **tf.global_variables**. They need to be explicitly added to the variable list of the Saver to restore in the future. Also, the update operations for them are in **tf.GraphKeys.UPDATE_OPS**, so **tf.GraphKeys.UPDATE_OPS** should be included into the control dependencies.
3.	In Stage I, our architecture is **Stage I Generator + Stage I Discriminators**. In Stage II, our architecure becomes **Stage I Generator + Stage II Generator + Stage II Discriminators**, but only the Stage II parameters are updated during optimization.
4.	We discover that using only uni-skip embedding of Skip-Thought generates images with greyish colors. So, we decide to use the 2400-dimensional Bi-Skip vector with the 200-dimensional vector generated from the embedding matrix in seq2seq.

<a id='Evaluation'></a>
#### Evaluation
Inception score是一種用來評量GAN產生圖片質量的方式，主要利用在Imagenet上Pretrain好的Inception Network來進行衡量，一個好的結果應有兩種特性:對一張圖所有Label的Entropy <img alt="$\mathrm{H}(y|x)$" src="svgs/d3dd77a58589f0c9238b0848cbf31029.png?invert_in_darkmode" align="middle" width="47.548545000000004pt" height="24.56552999999997pt"/>低，代表產生出來的圖很明確屬於某一個類別，代表越接近真實的圖片，而非一張完全不像任何東西，或一張由各種類別融合在一起的模糊圖片；對所有圖片的Entropy <img alt="$\mathrm{H}(y)$" src="svgs/0f5419de83f81bf7c04cc7e16b800c84.png?invert_in_darkmode" align="middle" width="33.639540000000004pt" height="24.56552999999997pt"/>高，代表所有圖片的多樣性大，沒有Mode Collapse的現象發生，我們希望在不同的Caption底下每張圖都應該是不盡相同的，就算Caption很接近，給定不同的初始Noise也應該有不同的圖片被產生。於是Inception score的算式如下:  

<p align="center"><img alt="$$\mathrm{Inception\;Score = exp}(\mathbb{E}_x\mathrm{KL}(p(y|x)\|p(y)))$$" src="svgs/780e4e4f40990534519e9aa06fbda8d2.png?invert_in_darkmode" align="middle" width="306.14265pt" height="16.376943pt"/></p>

Expanding the formula we get:
    
<p align="center"><img alt="$$\sum_x\sum_yp(y|x) \log p(y|x) - \sum_x\sum_yp(y|x) \log p(y) $$" src="svgs/190c7a2ea53f36abf3e2891588bcd7a2.png?invert_in_darkmode" align="middle" width="339.5172pt" height="38.38758pt"/></p>

前項是Negative Entropy of <img alt="$p(y|x)$" src="svgs/fc76db86ea6c427fdd05067ba4835daa.png?invert_in_darkmode" align="middle" width="43.50555pt" height="24.56552999999997pt"/>，我們希望越大越好;後項是Cross Entropy of <img alt="$p(y)$" src="svgs/c2f39888dc0b934028add620085faf36.png?invert_in_darkmode" align="middle" width="29.596545pt" height="24.56552999999997pt"/>，我們希望越小越好,所以整體上Inception Score越大表示結果越好。  

但實際上根據助教使用的程式碼，我們的理解是將圖片餵進VGG16，產生出來的Logits作為圖片的一種向量表示，使用Cosine Distance來計算產生結果和Ground Truth之間的差異，所以可以想像成Inception Score的<img alt="$\mathrm{H}(y|x)$" src="svgs/d3dd77a58589f0c9238b0848cbf31029.png?invert_in_darkmode" align="middle" width="47.548545000000004pt" height="24.56552999999997pt"/>，只不過越低越好；而另外計算所有圖片之間的Standard Deviation用來代替<img alt="$\mathrm{H}(y)$" src="svgs/0f5419de83f81bf7c04cc7e16b800c84.png?invert_in_darkmode" align="middle" width="33.639540000000004pt" height="24.56552999999997pt"/>，取倒數之後也是越低越好。

[Back to Top](#Top)

<a id='Result'></a>
### Result
Each row of flowers are generated from one of the 8 captions with different initial noise:  
1. the flower shown has yellow anther red pistil and bright red petals
2. this flower has petals that are yellow, white and purple and has dark lines
3. the petals on this flower are white with a yellow center
4. this flower has a lot of small round pink petals
5. this flower is orange in color, and has petals that are ruffled and rounded
6. the flower has yellow petals and the center of it is brown
7. this flower has petals that are blue and white
8. these white flowers have petals that start off white in color and end in a white towards the tips

<a id='Complex-Model'></a>
#### Complex Model
The original StackGAN paper use the complex generator discriminator. Nevertheless, in our experiments, we find out that the images generated are more unstable, the colors are more unnatural and the shapes are smoother. Furthermore, mode collapse happends easily after long enough training. The result is shown below:

After 50 epochs:

<p align="center"><img src="images/Complex_49.png"></p>

After 120 epochs:

<p align="center"><img src="images/Complex_119.png"></p>

<a id='Simple-Model'></a>
#### Simple Model
Based on our observations, the simple model always outperform the complex model.

After 5 epochs, the model learns the concept of color:

<p align="center"><img src="images/Simple_04.png"></p>

After 30 epochs, the shape of flower is roughtly formed:

<p align="center"><img src="images/Simple_29.png"></p>

After 120 epochs, ，部分文字的花已經非常逼真，但另外一些則呈現色塊混雜完全看不出是花的樣子:

<p align="center"><img src="images/Simple_119.png"></p>

第595個Epoch，可以發現原本就不錯的圖片變得更加逼真一些，但是原本壞掉的圖片仍沒有任何變好的跡象，可以觀察到第7,8個Caption的確是較複雜一些，Caption 7使用了形容詞子句來描述花瓣的顏色，而非常見的adj. + n.；Caption 8不但句子較長，還使用了Start off, End in等對花較細部的描述，且這幾個單字出現頻率並不高，因此學起來不這麼容易:

<p align="center"><img src="images/Simple_594.png"></p>

Flower evolution during Stage I training:

<p align="center"><img src="images/Result.gif"></p>

After 600 epochs of Stage II training, ，可以觀察到的確能有效的提升圖片的畫質，線條的細緻度和平滑度都更好，從Caption 2也可以觀察到Stage II的確有幫原圖添加細節的作用，本來在Stage I並沒有成功畫出Caption中Dark Lines的敘述；原本不成形的Caption 7也比較有花的特徵出現。
Stage II 第400個epoch:

<p align="center"><img src="images/StageII_994.png" width="600"></p>

[Back to Top](#Top)

<a id='Testing-Result'></a>
#### Testing Result
在助教提供的Testing Data上，我們Stage I最好的結果可以達到平均1.20分左右，我們發現即使加上Stage II來產生256×256的圖片，肉眼看起來好很多，但測出來精準度仍然差不多，甚至較差。  
下面我們分別取出在Inception Score上表現最好和表現最差的3組圖片，我們發現Inception Score和我們肉眼觀察的結果仍有一段差距，比如第148張圖，無論在Stage I或II怎麼看都不是花，分數卻非常高；第741張圖在低畫質的情況下看起來還可以，分數卻最差，到高畫質時才真正完全失真，這可能代表這張Generator產生的圖可能有我們肉眼看不出來的Noise，導致Inception模型無法正確判斷其內容，Stage II Generator也無法將他的畫質提升到更好。  

<a id='Top-3'></a>
**Top 3 Images**  

<p align="center"><img src="images/Top3.jpg" width="600"></p>

<a id='Bottom-3'></a>
**Bottom 3 Images**  
<p align="center"><img src="images/Bottom3.jpg" width="600"></p>

[Back to Top](#Top)

<a id='StageI-vs-StageII'></a>
**Stage I v.s. Stage II**  
下圖統計了每張圖片在Stage I和Stage II的分數，可以發現有些變高有些變低沒有一定的規律，深紅線和深藍分別是原始資料的移動平均：
<img src="images/Scores.PNG">

<a id='Cherry-Picking'></a>
**Cherry-Picked Images**  
<p align="center"><img src="images/Picked.jpg"></p>

<a id='Conclusion'></a>
### Conclusion
這次遇到比較大的困難有以下幾點，首先，StackGAN的實作有很多細節要注意，比如有些變數用**tf.random_normal**做初始化，有些用**tf.truncated_normal**等等，另外比較特別的是，無論是哪一種Layer，在架構中都是不加Bias的，我覺得這也符合在[GAN Tips]( https://github.com/soumith/ganhacks)中提到的必須避免Sparse Gradient，以增加Training的穩定性。  
在Training時，也很難透過Loss的變化來觀察模型的效果，除了一開始Discriminator的Loss會逐漸下降之外，後面無論Discriminator還是Generator的Loss都不會有明顯的變化，無法作為收斂的判斷依據；GAN產生的圖片也不一定Train越久就會越好，必須每個一段時間把圖片存下來才能找出最好的結果；此外，我們發現無論怎麼調整我們的參數、Embedding方式對分數的影響好像都不顯著，有可能影響GAN效能主要還是在模型架構的設計。  
我們一開始曾嘗試把Loss Function改為WGAN-GP，但發現Train不起來，參考了[原始論文]( https://arxiv.org/abs/1704.00028)才發現，WGAN-GP的Loss中是懲罰Generator產生的Distribution和真實圖片Distribution中間的一點，相當於一次只看一組資料，而Batch Normalization會加入一個Batch裡面每筆資料彼此之間的聯繫，導致Gradient Penalty失效，原論文建議可用Layer Normalization取代，但我們測試後還是Train不起來。  
雖然我們最後做出來的結果還算可以，但和原論文同樣Caption產生的圖片來比好像還差一段距離，可能是有小地方沒注意，或由於原論文使用char-CNN-RNN Text Encoders的關係。另外我們還可以改進的地方包括RNN模型，一開始沒想到像助教一樣用圖片當成RNN的Target，這邊感覺還有很多方法可以試；另外由於Flower Dataset每張圖其實都有Label，但這次我們並沒有使用到，若可以參考[AC-GAN]( https://arxiv.org/abs/1610.09585)增加Classification的Layer，我想也能更進一步改善最後的結果。  

[Back to Top](#Top)
