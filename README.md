# ganTest

mnistとcifar10をganで学習するコードです。

学習済みgeneratorの出力結果

mnist

![mnist_epoch300.png](https://github.com/YutaIzumi/ganTest/blob/master/img/mnist_epoch300.png)

cifar10(猫のみ)

![cat_epoch391.png](https://github.com/YutaIzumi/ganTest/blob/master/img/cat_epoch391.png)

## 環境構築
### 1.Anacondaをインストール
https://www.anaconda.com/distribution/#download-section

### 2.Anacondaプロンプトを起動、以下のコマンドを実行

#### tensorflowのインストール
※nvidiaのグラボがある場合は、

```conda install tensorflow-gpu```

※nvidiaのグラボが無い場合は、

```conda install tensorflow```

を実行して下さい。計算速度が全然違うのでgpu推奨です。

#### kerasのインストール
```conda install keras```

#### pipを使うとcudaやcudnnのバージョン違いで動かなくて悩むことになる可能性大です！condaでインストールして下さい。

## ganMnist.py
ファイルを実行するとmnistのテストデータを使って学習します。

コンソールにはランダムな入力に対するgeneratorの出力が表示されます。

最初は真っ黒ですが、学習が進むにつれて数字になっていくのが面白いｗ

## ganCifar10.py

mnistは28x28x1の画像データですが、cifar10は32x32x3の画像データです。

そのため重みや隠れ層の次元数がmnistの場合と変わります。そこだけ注意して下さい。

## ganCifar10_CatOnly.py

猫の画像のみ学習させます。 

## imageNetGan
### imageNet.py
imageNetから画像を取得するモジュール

参考：http://murayama.hatenablog.com/entry/2017/11/18/160818

32行目の画像IDを変えることでImageNetから様々な画像が一括取得できます。

```classes = {"img":"n02123159"}```

imgは画像を保存するフォルダ名です。n02123159は猫の画像のIDです。

画像のIDは以下で検索。

http://www.image-net.org/explore

### ImageResize.py
imgフォルダに保存した画像をリサイズしてnumpy配列で返すモジュール

ganImageNet.pyの中で使います。

### ganImageNet.py
imgフォルダに保存した画像を学習するモジュール

参考：http://13mzawa2.hateblo.jp/entry/2018/03/02/000901

### 使い方
```imageNetGan/img```に画像ファイルを保存

ganImageNet.pyを実行すると保存した画像に対して学習が開始します。

画像は64x64にリサイズされます。

りんごの画像を学習した結果。何となくりんごっぽい画像もありますが、精度はいまいちです。

![apple.png](https://github.com/YutaIzumi/ganTest/blob/master/img/apple.png)

## License
MIT
