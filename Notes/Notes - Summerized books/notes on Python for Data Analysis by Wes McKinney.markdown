### chpt 1: why python
"Part of Python’s success in scientific computing is the ease of integrating C, C++, and
FORTRAN code. Most modern computing environments share a similar set of legacy
FORTRAN and C libraries for doing linear algebra, optimization, integration, fast
Fourier transforms, and other such algorithms."

* why not just have interns rewrite these archaic libraries into python. python might be slower than C++ but I doubt the middling manning is fast

#### 1.3 Essential Python Libraries
* NumPy, short for Numerical Python, has long been a cornerstone of numerical com‐
puting in Python. It provides the data structures, algorithms, and library glue needed
for most scientific applications involving numerical data in Python.

* pandas provides high-level data structures and functions designed to make working
with structured or tabular data fast, easy, and expressive. It enables Python to be a powerful and productive data analysis environment. The primary objects in pandas that will be used in this book are the DataFrame,
a tabular, column-oriented data structure with both row and column labels, and the
*Series*, a one-dimensional labeled array object.

    ^^ DataFrame... that's litteral what I just learned in CMPT 318's first R lecture. -_-


* matplotlib is the most popular Python library for producing plots and other two dimensional data visualizations.

* SciPy is a collection of packages addressing a number of different standard problem
domains in scientific computing.
    ^^ im going to love scipy.optimize & scipy.integrate :)   wonder if ill being using this the most then i start machin learnign

* scikit-learn
Since the project’s inception in 2010, scikit-learn has become the premier generalpurpose machine learning toolkit for Python programmers.

      * Classification: SVM, nearest neighbors, random forest, logistic regression, etc.
      * Regression: Lasso, ridge regression, etc.
      * Clustering: k-means, spectral clustering, etc.
      * Dimensionality reduction: PCA, feature selection, matrix factorization, etc.
      * Model selection: Grid search, cross-validation, metrics
      * Preprocessing: Feature extraction, normalization

### chpt 2: Python Language Basics
[my fav TL;DR video](https://www.youtube.com/watch?v=N4mEzFDjqtA)

* a // b Floor-divide a by b, dropping any fractional remainder
* a ** b Raise a to the b power
* a is b True if a and b reference the same Python object
* a is not b True if a and b reference diﬀerent Python objects

* datetime is a important OBJ. I will likely forgot its name

`pass` is like `continue` from java

Ternary in python
``** 'Non-negative' if x >= 0 else 'Negative' **``



### chpt 3:Built-in Data Structures, Functions, and Files

* Tuple is a fixed-length, immutable sequence of Python objects. The easiest way to
create one is with a comma-separated sequence of values

* List In contrast with tuples, lists are variable-length and their contents can be modified
in-place. You can define them using square brackets [] or using the list type function
  * lists work well with set theory... good thing i paid attension back in MACM 101


returning multiple values

    def f():
      a = 5
      b = 6
      c = 7
      return a, b, c

    a, b, c = f()

damm I wish JS had this. nice way to confuse a code reviewer

#### functions as objects

    import re

    def clean_strings(strings):
      result = []
      for value in strings:
          value = value.strip()
          value = re.sub('[!#?]', '', value)
          value = value.title()
          result.append(value)
        return result

    In [173]: clean_strings(states)
    Out[173]:
    ['Alabama',
    'Georgia',
    'Georgia',
    'Georgia',
    'Florida',
    'South Carolina',
    'West Virginia']


#### functions as parameter

    def apply_to_list(some_list, f):
      return [f(x) for x in some_list]

    ints = [4, 0, 1, 5, 6]
    apply_to_list(ints, lambda x: x * 2)


#### Currying

Currying is computer science jargon (named after the mathematician Haskell Curry)
that means deriving new functions from existing ones by partial argument applica‐
tion. For example, suppose we had a trivial function that adds two numbers together:
      def add_numbers(x, y):
        return x + y

Using this function, we could derive a new function of one variable, add_five, that
adds 5 to its argument:

      add_five = lambda y: add_numbers(5, y)
The second argument to add_numbers is said to be curried. There’s nothing very fancy
here


#### Generators //TODO... ya java comments in python notes


#### Exceptions

    f = open(path, 'w')
    try:
      write_to_file(f)
    except:
      print('Failed')
    else:
      print('Succeeded')
    finally:
      f.close()

#### 3.3 Files and the Operating System
##### omit cuz im lazy ( ill have to use google for this anyways)





### chpt 4: NumPy Basics

#### 2d array

    data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
     arr2 = np.array(data2)
     # 2*4 matrix

#### indexing and slicing

    Out[61]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    In [63]: arr[5:8]
    Out[63]: array([5, 6, 7])

    In [66]: arr_slice = arr[5:8]
    In [68]: arr_slice[1] = 12345
    In [69]: arr
    Out[69]: array([ 0, 1, 2, 3, 4, 12, 12345, 12, 8, 9])

    In [70]: arr_slice[:] = 64
    In [71]: arr
    Out[71]: array([ 0, 1, 2, 3, 4, 64, 64, 64, 8, 9])


#### 3d array

    In [74]: arr2d[0][2]
    Out[74]: 3
    In [75]: arr2d[0, 2]
    Out[75]: 3
    # same thing?

    arr2d[:2]
    #outputs first 2 rows

    arr2d[:2, 1:]
    # end at 2nd row, start at 1st column

![](matrixSlice.png)

#### boolean indexing //TODO

#### fancy indexing

      In [119]: arr
      Out[119]:
      array([[ 0., 0., 0., 0.],
            [ 1., 1., 1., 1.],
            [ 2., 2., 2., 2.],
            [ 3., 3., 3., 3.],
            [ 4., 4., 4., 4.],
            [ 5., 5., 5., 5.],
            [ 6., 6., 6., 6.],
            [ 7., 7., 7., 7.]])
To select out a subset of the rows in a particular order, you can simply pass a list or ndarray of integers specifying the desired order:

          In [120]: arr[[4, 3, 0, 6]]
          Out[120]:
          array([[ 4., 4., 4., 4.],
                [ 3., 3., 3., 3.],
                [ 0., 0., 0., 0.],
                [ 6., 6., 6., 6.]])


#### 4.3 Array-Oriented Programming with Arrays pg 108 //TODO


#### 4.4 File Input and Output with Arrays

* NumPy to load and save arrays with .npy format
* multiple arrays in dictionary like formate in a .npz
        In [217]: arch = np.load('array_archive.npz')
        In [218]: arch['b']
        Out[218]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])



#### 4.5 Linear Algebra

title says it all


#### 4.6 Pseudorandom Number Generation

TL;DR use NumPy for random number generation



### chpt: 5 pandas

[pandas cheat sheet](https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf)

[pandas cheat sheet 2](https://www.dataquest.io/blog/large_files/pandas-cheat-sheet.pdf)






### chpt 6: Data Loading, Storage, and File Formats

some parsing functions in pandas

  * read_csv
  * read_json
  * read_excel




    example of reading data
      pd.read_table('examples/ex1.csv', sep=',')

    no header  
      In [13]: pd.read_csv('examples/ex2.csv', header=None)

    custom header   
      In [14]: pd.read_csv('examples/ex2.csv', names=['a', 'b', 'c', 'd', 'message'])



#### 6.3 Interacting with Web APIs

    better to google this


#### 6.4 Interacting with Databases

    this parts appear to be useless. doesnt say anything about locating a DB, or creating one. only about creating a query....

### chpt 7: data cleaning and prep



  ##### Filtering Out Missing Data

    data = pd.Series([1, NA, 3.5, NA, 7])
    data.dropna() // yay! missing data is gone


when

      data = pd.DataFrame( ....... )

      data.drop_duplicates() // returns DataFrame without duplicates



will add column 'animal'

        data['animal'] = lowercased.map(meat_to_animal)



#### Replacing Values

      data.replace(-999, np.nan) // done!!
      data.replace([-999, -1000], [np.nan, 0]) // as dual lists
      data.replace({-999: np.nan, -1000: 0}) // as OBJ




### chpt 8: Data Wrangling: Join, Combine, and Reshape


#### 8.1 Hierarchical Indexing

    data = pd.Series(np.random.randn(9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 3, 1, 2, 2, 3]])

    a 1 -0.204708
      2 0.478943
      3 -0.519439
    b 1 -0.555730
      3 1.965781
    c 1 1.393406
      2 0.092908
    d 2 0.281746
    2213 0.769023
//----------------- yes I use this as a formatting device

    In [12]: data['b']
    Out[12]:
      1 -0.555730
      3 1.965781
    //-------------------------------
    In [14]: data.loc[['b', 'd']]
    Out[14]:
    b 1 -0.555730
      3 1.965781
    d 2 0.281746
      3 0.769023
      //-------------
      In [16]: data.unstack()
      Out[16]:
      1            2         3
    a -0.204708 0.478943 -0.519439
    b -0.555730 NaN 1.965781

lets combine

        frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
        index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],  
        columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])

        Ohio Colorado
        Green Red Green
    a 1      0 1 2
      2      3 4 5
    b 1      6 7 8
      2    9 10 11


#### 8.2 Combining and Merging Datasets

> not a notes friendly section. im omiting this :)



#### 8.3  Reshaping and Pivoting

      pd.DataFrame(np.arange(6).reshape((2, 3))) // 2 rows, 3 columns


* Pivoting is just confusing...



### chpt 9: Plotting and Visualization

>  import matplotlib.pyplot as plt

* diagonal line is made like:

        data = np.arange(10)
        plt.plot(data)


* to save

      plt.savefig('figpath.svg')

* graph

      df = pd.DataFrame(np.random.randn(10, 4).cumsum(0),
      columns=['A', 'B', 'C', 'D'],
      index=np.arange(0, 100, 10))

      df.plot() // hey! look a invisible graph



* Bar Plots are similar
      df = myDataFrameOBJ
      df.plot.bar()


### chpt 10: Data Aggregation and Group Operations


##### Optimized groupby methods
* count Number of non-NA values in the group
* sum Sum of non-NA values
* mean Mean of non-NA values
* median Arithmetic median of non-NA values
* std, var Unbiased (n – 1 denominator) standard deviation and variance
* min, max Minimum and maximum of non-NA values
* prod Product of non-NA values
* first, last  First and last non-NA values


      grouped = df.groupby('key1')
      grouped.describe() // basic but useful stats about the DataFrame


add column

    tips['tip_pct'] = tips['tip'] / tips['total_bill']


cut our data into quartiles

        In [82]: frame = pd.DataFrame({'data1': np.random.randn(1000),
        ....: 'data2': np.random.randn(1000)})
        In [83]: quartiles = pd.cut(frame.data1, 4)
        In [84]: quartiles[:10]
        Out[84]:
        0 (-1.23, 0.489]
        1 (-2.956, -1.23]
        2 (-1.23, 0.489]
        3 (0.489, 2.208]
        4 (-1.23, 0.489]
        5 (0.489, 2.208]
        6 (-1.23, 0.489]
        7 (-1.23, 0.489]
        8 (0.489, 2.208]
        9 (0.489, 2.208]
        Name: data1, dtype: category
        Categories (4, interval[float64]): [(-2.956, -1.23] < (-1.23, 0.489] < (0.489, 2.
        208] < (2.208, 3.928]]


pivot tables

      In [130]: tips.pivot_table(index=['day', 'smoker'])
      Out[130]:
                size tip tip_pct total_bill
      day smoker
      Fri No      2.250000 2.812500 0.151650 18.420000
          Yes     2.066667 2.714000 0.174783 16.813333
      Sat No      2.555556 3.102889 0.158048 19.661778
          Yes     2.476190 2.875476 0.147906 21.276667
      Sun No      2.929825 3.167895 0.160113 20.506667
          Yes     2.578947 3.516842 0.187250 24.120000
      Thur No     2.488889 2.673778 0.160298 17.113111
          Yes     2.352941 3.030000 0.163863 19.190588




### chpt 11: time series

>   from datetime import datetime


      now = datetime.now()
      datetime.datetime(2017, 9, 25, 14, 5, 52, 72973)

      //----------------------------------------------

      In [13]: now.year, now.month, now.day
      Out[13]: (2017, 9, 25)


      In [14]: delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
      In [15]: delta
      Out[15]: datetime.timedelta(926, 56700)
      In [16]: delta.days
      Out[16]: 926
      In [17]: delta.seconds
      Out[17]: 56700


#### Converting Between String and Datetime

      In [22]: stamp = datetime(2011, 1, 3)
      In [23]: str(stamp)
      Out[23]: '2011-01-03 00:00:00'
      In [24]: stamp.strftime('%Y-%m-%d')
      Out[24]: '2011-01-03'




      In [25]: value = '2011-01-03'
      In [26]: datetime.strptime(value, '%Y-%m-%d')
      Out[26]: datetime.datetime(2011, 1, 3, 0, 0)

* NaT (Not a Time) is pandas’s null value for timestamp data.

build list of dates between two points

       index = pd.date_range('2012-04-01', '2012-06-01')

      //alt way
              pd.date_range(start='2012-04-01', periods=20)


#### Base time series frequencies are like ->  freq='BM'
* BM is Business Month End
* MS is month begin
* BMS business month begin




### chpt 12: Advanced pandas


> ive looked over this chapter twice, nothing interesting that i want to remember -_-






### chpt 13: Introduction to Modeling Libraries in Python



#### [patsy formulas](https://patsy.readthedocs.io/en/v0.1.0/formulas.html)

#### Estimating Linear Models

      import statsmodels.api as sm
      import statsmodels.formula.api as smf


##### i dont understand the stats in this to write worthwhile notes (looks like 3 to 4th year stuff)


          In [70]: model = sm.OLS(y, X)
          The model’s fit method returns a regression results object containing estimated
          model parameters and other diagnostics:
          In [71]: results = model.fit()
          In [72]: results.params
          Out[72]: array([ 0.1783, 0.223 , 0.501 ])
          The summary method on results can print a model detailing diagnostic output of the
          model:
          In [73]: print(results.summary())
          OLS Regression Results
          ==============================================================================
          Dep. Variable: y
           R-squared: 0.430
          Model: OLS Adj.
           R-squared: 0.413
          Method: Least Squares
          F-statistic: 24.42
          Date: Mon, 25 Sep 2017
           Prob (F-statistic): 7.44e-12
          Time: 14:06:15
          Log-Likelihood: -34.305
          No. Observations: 100
          AIC: 74.61
          Df Residuals: 97
          BIC: 82.42
          Df Model: 3
          Covariance Type: nonrobust
          ==============================================================================
          coef std err t P>|t| [0.025 0.975]
          ------------------------------------------------------------------------------
          x1 0.1783 0.053 3.364 0.001 0.073 0.283
          x2 0.2230 0.046 4.818 0.000 0.131 0.315
          x3 0.5010 0.080 6.237 0.000 0.342 0.660
          ==============================================================================
          Omnibus: 4.662
           Durbin-Watson: 2.201
          Prob(Omnibus): 0.097
          Jarque-Bera (JB): 4.098
          Skew: 0.481
          Prob(JB): 0.129
          Kurtosis: 3.243
          Cond. No. 1.74
          ==============================================================================










#### 13.4 Introduction to scikit-learn

pg 397 - 401  this is useless...


      In [86]: train = pd.read_csv('datasets/titanic/train.csv')
      In [87]: test = pd.read_csv('datasets/titanic/test.csv')

      train.isnull().sum() // list of how many nulls from each column

      In [91]: impute_value = train['Age'].median()
      In [92]: train['Age'] = train['Age'].fillna(impute_value)
      //'fix' the nulls by replacing with M
      In [93]: test['Age'] = test['Age'].fillna(impute_value)



      //-----------------------------------------
      In [96]: predictors = ['Pclass', 'IsFemale', 'Age']
      In [97]: X_train = train[predictors].values
      In [98]: X_test = test[predictors].values
      In [99]: y_train = train['Survived'].values
      In [100]: X_train[:5]

      Out[100]:
      array([[ 3., 0., 22.],
      [ 1., 1., 38.],
      [ 3., 1., 26.],
      [ 1., 1., 35.],
      [ 3., 0., 35.]])
      In [101]: y_train[:5]
      Out[101]: array([0, 1, 1, 1, 0])

      then,

      In [102]: from sklearn.linear_model import LogisticRegression
      In [103]: model = LogisticRegression()

      Similar to statsmodels, we can fit this model to the training data using the model’s fit
      method:

      In [104]: model.fit(X_train, y_train)
      Out[104]:
      LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
      intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
      penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
      verbose=0, warm_start=False)

      Now, we can form predictions for the test dataset using model.predict:

      In [105]: y_predict = model.predict(X_test)
      In [106]: y_predict[:10]
      Out[106]: array([0, 0, 0, 0, 1, 0, 1, 0, 1, 0])






####  13.5 Continuing Your Education

  This book is focused especially on data wrangling, but there are many others dedica‐
  ted to modeling and data science tools. Some excellent ones are:

  * Introduction to Machine Learning with Python by Andreas Mueller and Sarah
  Guido (O’Reilly)
  * Python Data Science Handbook by Jake VanderPlas (O’Reilly)
  * Data Science from Scratch: First Principles with Python by Joel Grus (O’Reilly)
  * Python Machine Learning by Sebastian Raschka (Packt Publishing)
  * Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien
  Géron (O’Reilly)


  While books can be valuable resources for learning, they can sometimes grow out of
  date when the underlying open source software changes. It’s a good idea to be familiar
  with the documentation for the various statistics or machine learning frameworks to
  stay up to date on the latest features and API.



### chpt 14: Data Analysis Examples

#### best to stick to the book its self for this chapter. pg: 403 (421 / 541)







### advanced NumPy
#### sorting  

first column only

        In [163]: arr = np.random.randn(3, 5)
        In [164]: arr
        Out[164]:
        array([[-0.3318, -1.4711, 0.8705, -0.0847, -1.1329],
        [-1.0111, -0.3436, 2.1714, 0.1234, -0.0189],
        [ 0.1773, 0.7424, 0.8548, 1.038 , -0.329 ]])
        In [165]: arr[:, 0].sort() # Sort first column values in-place
        In [166]: arr
        Out[166]:
        array([[-1.0111, -1.4711, 0.8705, -0.0847, -1.1329],
        [-0.3318, -0.3436, 2.1714, 0.1234, -0.0189],
        [ 0.1773, 0.7424, 0.8548, 1.038 , -0.329 ]])


sort items inside each row

      All of these sort methods take an axis argument for sorting the sections of data along
      the passed axis independently:
          In [171]: arr = np.random.randn(3, 5)
          In [172]: arr
          Out[172]:
          array([[ 0.5955, -0.2682, 1.3389, -0.1872, 0.9111],
          [-0.3215, 1.0054, -0.5168, 1.1925, -0.1989],
          [ 0.3969, -1.7638, 0.6071, -0.2222, -0.2171]])
          In [173]: arr.sort(axis=1)
          In [174]: arr
          Out[174]:
          array([[-0.2682, -0.1872, 0.5955, 0.9111, 1.3389],
          [-0.5168, -0.3215, -0.1989, 1.0054, 1.1925],
          [-1.7638, -0.2222, -0.2171, 0.3969, 0.6071]])
