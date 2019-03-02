# ganTest

# 環境構築
1.Anacondaをインストール

2.Anacondaプロンプトを起動

3.以下のコマンドを実行

### tensorflowのインストール
conda install tensorflow-gpu

※nvidiaのグラボが無い場合は、

conda install tensorflow

を実行して下さい。

### kerasのインストール（一応）
conda install keras

### pipを使うとバージョン違いで動かなくて悩むことになる可能性大です！

# ganMnist.py
ファイルを実行するとmnistのテストデータを使って学習します。

コンソールにはランダムな入力に対するgeneratorの出力が表示されます。

最初は真っ黒ですが、学習が進むにつれて数字になっていくのが面白いｗ
