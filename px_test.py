import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
#Seriesは一次元、Dataframeは二次元のデータを収納するために使われる

from numpy.random import normal
#正規分布乱数
import xlwings as xw
import openpyxl as xl

#wb = xl.load_workbook("px_test.xlsm", data_only="True")
wb = xl.load_workbook("px_test.xlsm")
ws = wb["Sheet1"]



def create_dataset(num):
  xs = np.linspace(0, 1, num)
  ts = np.linspace(0, 1, num)
  i = 0
  for i in range(num):
    xs[i] = ws["A{}".format(i+1)].value
    ts[i] = ws["B{}".format(i+1)].value
  return xs, ts



N = 10 # データ数
xs, ts = create_dataset(N)

DataFrame({'x': xs, 't': ts})



fig = plt.figure(figsize=(6, 4))     #figure()は図。グラフ描画の台紙と思えばいい。figuresizeは台紙の縦横比を指定。
subplot = fig.add_subplot(1, 1, 1)    #(1, 1, 1)とはfigure()の領域内での一行目一列目の一番目という意味。上書きしたらグラフが重なったりする。
subplot.tick_params(axis='x', labelsize=12)     #params()は各軸の設定。 labelsizeは目盛りの文字サイズ
subplot.tick_params(axis='y', labelsize=12)
subplot.set_xlim(-0.05, 1.05)    #それぞれxy軸の上下限を設定。
subplot.set_ylim(-1.5, 1.5)
_ = subplot.scatter(xs, ts, marker='o', color='blue')    #散布図を描画する関数 scatter(x軸配列, y軸配列) あとは任意で追加。markerは形状を指定。
#pythonでは _(アンダースコア) も変数名として使用できる。
plt.savefig("test.png")



def resolve(xs, ts, m):
  phi = np.array([[x**k for k in range(m+1)] for x in xs])     #xのk乗の(m+1)×(xs)行列。 (m+1)側がk、xs側がxでループを回している
  tmp = np.linalg.inv(np.dot(phi.T, phi))      #linalg.inv()が逆行列。dot()が行列の積。つまり、phiの転置とphiの積の逆行列。
  ws = np.dot(np.dot(tmp, phi.T), ts)          #(tmpとphiの転置の積)と(ts)の積
                                               #つまりphi=Φ、tmp=逆行列項、ws=w

  def f(x):                                    #関数のネスト
    y = 0
    for i, w in enumerate(ws):                 #enumerate()はforを回しつつインデックス番号を取得出来るらしい。(f()はΣとに番号いるか..?)
      y += w * x**i                            #(2.16)のΣの式。つまり今回導出しようとしているM次多項式
    return y

  return f, ws



def rms_error(xs, ts, f):
  err = 0.5 * np.sum((f(xs) - ts)**2)           #二乗誤差
  return np.sqrt(2 * err / len(xs))             #平方平均二乗誤差



def show_result(subplot, xs, ts, m):
  f, _ = resolve(xs, ts, m)                                      #近似多項式f()と、多項式の係数wを返す内部定義関数
  subplot.tick_params(axis='x', labelsize=12)                    #グラフの軸の設定。軸のフォントサイズ設定
  subplot.tick_params(axis='y', labelsize=12)
  subplot.set_xlim(-0.05, 1.05)                                  #グラフの上下限設定
  subplot.set_ylim(-1.5, 1.5)
  subplot.set_title('M={}'.format(m), fontsize=14)               #グラフのタイトル設定。format()は書式化の関数。

  # トレーニングセットを表示
  subplot.scatter(xs, ts, marker='o', color='blue', label=None) #scatter()散布図の描画

  linex = np.linspace(0, 1, 100)
  liney = f(linex)
  label = 'E(RMS)={:.2f}'.format(rms_error(xs, ts, f))          #平方平均二乗誤差を小数点以下二桁までラベルで表示
  subplot.plot(linex, liney, color='red', label=label)          #f(x)を赤線で描画
  subplot.legend(loc=1, fontsize=14)                            #凡例の表示(-- E(RMS)のところ)




fig = plt.figure(figsize=(12, 8.5))                    #グラフの縦横比
fig.subplots_adjust(wspace=0.3, hspace=0.3)            #グラフ同士の間隔
for i, m in enumerate([1, 2, 3, 4]):                   #0, 1, 3, 9でforを回す
  subplot = fig.add_subplot(2, 2, i+1)                 #二行二列で座標を導入
  show_result(subplot, xs, ts, m)                      #正解ラベル、sin2π、近似多項式を描画する内部定義関数
plt.savefig("test_result.png")
rate([0, 1, 3, 9]):


