# ganTest

mnistとCifar10をganで学習するコードです。

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

## ganCifar10

mnistは28x28x1の画像データですが、cifar10は32x32x3の画像データです。

そのため重みや隠れ層の次元数がmnistの場合と変わります。そこだけ注意して下さい。

## License
MIT
