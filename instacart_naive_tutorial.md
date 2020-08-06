# This is a simple and naive approach for a big matrix decomposition

### 이 노트북에서는 instacart dataset의 사용자별 선호 상품을 SVD로 예측해본다.
### 다음과 같은 문제가 예상되고, 그에 대한 단순한 해결책을 제시한다.
    1. 매우 큰 차원의 데이터에 대한 decomposition을 어떻게 처리해야 하나?
    2. sparsity가 높은 행렬에 대한 SVD를 어떻게 처리해야 하나?

## data set: instacart(from kaggle)
    1. aisles.csv: 총 134개의 품목 섹션 데이터(대형 마트의 aisels = 통로, 물건 사이의 통로)
    2. order_products__prior.csv: 총 3243449개의 주문 정보. 주문자, 품목, 과거 주문 여부(0이면 신규주문)
    3. departments.csv: 코너 이름(냉동식품, 기타, 베이커리 등)
    4. products.csv: 총 상품 종류(49688개)
    5. orders.csv: 
    6. sample_submission.csv:
    7. order_products__train.csv: 학습용


## 1. simple EDA
    1. 기초통계량 및 분포 확인
    2. correlation between items


```python
# import block
import pandas as pd
import numpy as np
import zipfile as zp
import os
```


```python
# local path and file names
base_path = '/home/ssum/바탕화면/experiments/instacart/'
file_names = os.listdir(base_path)
```


```python
file_names
```




    ['order_products__prior.csv.zip',
     'aisles.csv.zip',
     'departments.csv.zip',
     'products.csv.zip',
     'orders.csv.zip',
     'sample_submission.csv.zip',
     'order_products__train.csv.zip']




```python
# order_products
zip_file = zp.ZipFile(base_path+file_names[0])
order_product__prior = pd.read_csv(zip_file.open('order_products__prior.csv'),sep=',')
order_product__prior
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>33120</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>28985</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>9327</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>45918</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>30035</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>32434484</td>
      <td>3421083</td>
      <td>39678</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <td>32434485</td>
      <td>3421083</td>
      <td>11352</td>
      <td>7</td>
      <td>0</td>
    </tr>
    <tr>
      <td>32434486</td>
      <td>3421083</td>
      <td>4600</td>
      <td>8</td>
      <td>0</td>
    </tr>
    <tr>
      <td>32434487</td>
      <td>3421083</td>
      <td>24852</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <td>32434488</td>
      <td>3421083</td>
      <td>5020</td>
      <td>10</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>32434489 rows × 4 columns</p>
</div>




```python
# order_products_train
zip_file = zp.ZipFile(base_path+file_names[6])
order_product__train = pd.read_csv(zip_file.open('order_products__train.csv'),sep=',')
order_product__train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>product_id</th>
      <th>add_to_cart_order</th>
      <th>reordered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>49302</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>11109</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1</td>
      <td>10246</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>49683</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>43633</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1384612</td>
      <td>3421063</td>
      <td>14233</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1384613</td>
      <td>3421063</td>
      <td>35548</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1384614</td>
      <td>3421070</td>
      <td>35951</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1384615</td>
      <td>3421070</td>
      <td>16953</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1384616</td>
      <td>3421070</td>
      <td>4724</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1384617 rows × 4 columns</p>
</div>




```python
# aisels
zip_file = zp.ZipFile(base_path+file_names[1])
aisles = pd.read_csv(zip_file.open('aisles.csv'),sep=',')
aisles
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>aisle_id</th>
      <th>aisle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>prepared soups salads</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>specialty cheeses</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>energy granola bars</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>instant foods</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>marinades meat preparation</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>129</td>
      <td>130</td>
      <td>hot cereal pancake mixes</td>
    </tr>
    <tr>
      <td>130</td>
      <td>131</td>
      <td>dry pasta</td>
    </tr>
    <tr>
      <td>131</td>
      <td>132</td>
      <td>beauty</td>
    </tr>
    <tr>
      <td>132</td>
      <td>133</td>
      <td>muscles joints pain relief</td>
    </tr>
    <tr>
      <td>133</td>
      <td>134</td>
      <td>specialty wines champagnes</td>
    </tr>
  </tbody>
</table>
<p>134 rows × 2 columns</p>
</div>




```python
# aisels
zip_file = zp.ZipFile(base_path+file_names[3])
products = pd.read_csv(zip_file.open('products.csv'),sep=',')
products
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_id</th>
      <th>product_name</th>
      <th>aisle_id</th>
      <th>department_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Chocolate Sandwich Cookies</td>
      <td>61</td>
      <td>19</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>All-Seasons Salt</td>
      <td>104</td>
      <td>13</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Robust Golden Unsweetened Oolong Tea</td>
      <td>94</td>
      <td>7</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Smart Ones Classic Favorites Mini Rigatoni Wit...</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Green Chile Anytime Sauce</td>
      <td>5</td>
      <td>13</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>49683</td>
      <td>49684</td>
      <td>Vodka, Triple Distilled, Twist of Vanilla</td>
      <td>124</td>
      <td>5</td>
    </tr>
    <tr>
      <td>49684</td>
      <td>49685</td>
      <td>En Croute Roast Hazelnut Cranberry</td>
      <td>42</td>
      <td>1</td>
    </tr>
    <tr>
      <td>49685</td>
      <td>49686</td>
      <td>Artisan Baguette</td>
      <td>112</td>
      <td>3</td>
    </tr>
    <tr>
      <td>49686</td>
      <td>49687</td>
      <td>Smartblend Healthy Metabolism Dry Cat Food</td>
      <td>41</td>
      <td>8</td>
    </tr>
    <tr>
      <td>49687</td>
      <td>49688</td>
      <td>Fresh Foaming Cleanser</td>
      <td>73</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
<p>49688 rows × 4 columns</p>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
