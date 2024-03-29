import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import tree
import graphviz
import numpy as np

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def Visual(X,Y,Z):
    X = X_train['X'].tolist() + X_test['X'].tolist()
    Y = X_train['Y'].tolist() + X_test['Y'].tolist()
    Z = df_train['NTG'].tolist() + y_pred.tolist()
    # print(X)
    # print(Y)
    # print(Z)

    ####визуализация прогноза

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    C = plt.tricontour(X, Y, Z, colors='black', linewidths=1)
    plt.tricontourf(X, Y, Z)
    plt.clabel(C, inline=1, fontsize=10)
    plt.show()

df_test = pd.read_csv('Empty_part.csv')
df_train = pd.read_csv('Training_wells.csv')

df_test = df_test.drop(['Well'], axis=1)
df_train = df_train.drop(['Well'], axis=1)

print(df_train)
print(df_test)

df = df_train
x = df['X']
y = df['Y']

fig, ax = plt.subplots()

ax.scatter(x, y,c = 'red')    #  цвет точек

df = df_test
x = df['X']
y = df['Y']

ax.scatter(x, y, c = 'green')    #  цвет точек
ax.set_facecolor('black')

fig.set_figwidth(10)     #  ширина и
fig.set_figheight(10)    #  высота "Figure"

plt.show()

plt.style.use('fivethirtyeight')
z=df_train['NTG']
#z=sorted(z)
print(z)
plt.hist(z, bins = 150, edgecolor = 'k');
plt.xlabel('NTG'); plt.ylabel('Кол-во повторений');
plt.title('Гистограмма частотного анализа');

##TODO почистить значения, сто проц убрать пик 235	903	0,5625

df = pd.concat([df_train, df_test])#объединяем фреймы
df['NTG'].fillna(0.0, inplace=True)#зануляем пропуски для построение поверхности
print(df)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
X = df['X']
Y = df['Y']
Z = df['NTG']



fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C=plt.tricontour(X,Y,Z,colors='black',linewidths=1)
plt.tricontourf(X,Y,Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()

##прогноз

X_train= df_train[['X','Y']]
y_train= df_train['NTG']
X_test= df_test[['X','Y']]

print(X_train)
print(y_train)
print(X_test)
# линейная регрессия
# regressor = LinearRegression()
# regressor.fit(X_train, y_train) #training the algorithm
# y_pred = regressor.predict(X_test)
#
# print(y_pred)

# ######
# ######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# %matplotlib inline
import random

# set seed to reproducible
# random.seed(1)
# попробовать заунилть координаты
x = X_train['X'].tolist()
y = X_train['Y'].tolist()
z = df_train['NTG'].tolist()
fig = plt.figure(figsize=(10, 6))
ax = axes3d.Axes3D(fig)
ax.scatter3D(x, y, z, c='r')

x_grid = np.linspace(201, 246, 46)
y_grid = np.linspace(901, 930, 30)

B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
Z = np.zeros((len(x), len(y)))

import scipy as sp
import scipy.interpolate

spline = sp.interpolate.Rbf(x, y, z, function='thin_plate')

Z = spline(B1, B2)

np.savetxt('B1.csv', B1, delimiter=',')
np.savetxt('B2.csv', B2, delimiter=',')
np.savetxt('Z.csv', Z, delimiter=',')

df_exportSpline = pd.read_csv('Empty_part.csv')
x = df_exportSpline['X'].tolist()
y = df_exportSpline['Y'].tolist()

# for xy in df_exportSpline:
#     for b1 in B1:
#         if b1==xy['X']:
#             for b2 in B2:
#                 if b2 ==xy['Y']:
#                     z_arr.append()

print(len(B1))
count = 0
print(Z[29][45])
arr = []
for i in x:
    for j in y:
        arr.append(Z[j - 901][i - 201])
print(arr)

Visual(x, y, z)
VisualF(B1, B2, Z)

print(B1)
print(B2)
print(Z)

# Visual(B1,B2,Z)
fig = plt.figure(figsize=(10, 6))
ax = axes3d.Axes3D(fig)
ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z, alpha=0.2)
ax.scatter3D(x, y, z, c='r')

# for k in range(1,len(df_exportSpline)):
#     for i in range(0,len(B1)):
#         for j in range(0,len(B2)):
#             if(df_exportSpline[1][k]==B1[i][0] and df_exportSpline[2][k]==B2[0][j]):
#                 count+=1
# print(df_exportSpline[0][0])

print(B1[0][0])
print(B2[0][0])
print(count)
# ########
# ########
# кнн регрессия
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X_train, y_train)
KNeighborsRegressor(...)
y_pred=neigh.predict(X_test)

print(y_pred)
###преобразование для визуализации
X_train
X_test


y_pred
print (y_train)
X=X_train['X'].tolist() + X_test['X'].tolist()
Y=X_train['Y'].tolist() + X_test['Y'].tolist()
Z=df_train['NTG'].tolist() + y_pred.tolist()
print(X)
print(Y)
print(Z)
####визуализация прогноза

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C=plt.tricontour(X,Y,Z,colors='black',linewidths=1)
plt.tricontourf(X,Y,Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()

# ######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# %matplotlib inline
import random
# set seed to reproducible
# random.seed(1)
# попробовать заунилть координаты
x = X_train['X'].tolist()
y = X_train['Y'].tolist()
z = df_train['NTG'].tolist()
fig = plt.figure(figsize=(10,6))
ax = axes3d.Axes3D(fig)
ax.scatter3D(x,y,z, c='r')


x_grid = np.linspace(min(x)+1, max(x)-1, len(x))
y_grid = np.linspace(min(y)+1, max(y)-1, len(y))

B1, B2 = np.meshgrid(x_grid, y_grid, indexing='xy')
Z = np.zeros(( len(x), len(y) ))

import scipy as sp
import scipy.interpolate
spline = sp.interpolate.Rbf(x,y,z,function='thin_plate')

Z = spline(B1,B2)

Visual(x,y,z)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(B1, B2, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C = plt.contour(B1, B2, Z, colors='black', linewidths=1)
plt.contourf(B1, B2, Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()
# Visual(B1,B2,Z)
fig = plt.figure(figsize=(10,6))
ax = axes3d.Axes3D(fig)
ax.plot_wireframe(B1, B2, Z)
ax.plot_surface(B1, B2, Z,alpha=0.2)
ax.scatter3D(x,y,z, c='r')


# ########

# древесная регерессия
from sklearn.tree import DecisionTreeRegressor

# create a regressor object
regressor = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

###преобразование для визуализации
X_train
X_test

y_pred
print(y_train)
X = X_train['X'].tolist() + X_test['X'].tolist()
Y = X_train['Y'].tolist() + X_test['Y'].tolist()
Z = df_train['NTG'].tolist() + y_pred.tolist()
# print(X)
# print(Y)
# print(Z)

####визуализация прогноза

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C = plt.tricontour(X, Y, Z, colors='black', linewidths=1)
plt.tricontourf(X, Y, Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()


## метод рандомной древесины
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
###преобразование для визуализации
X_train
X_test
y_pred
print (y_train)
X=X_train['X'].tolist() + X_test['X'].tolist()
Y=X_train['Y'].tolist() + X_test['Y'].tolist()
Z=df_train['NTG'].tolist() + y_pred.tolist()
# print(X)
# print(Y)
# print(Z)

####визуализация прогноза

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C=plt.tricontour(X,Y,Z,colors='black',linewidths=1)
plt.tricontourf(X,Y,Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()


# метод опорных векторв
from sklearn import svm
support = svm.SVC()
support.fit(X_train, y_train)
pred = support.predict(X_test)


###преобразование для визуализации
X_train
X_test


y_pred
print (y_train)
X=X_train['X'].tolist() + X_test['X'].tolist()
Y=X_train['Y'].tolist() + X_test['Y'].tolist()
Z=df_train['NTG'].tolist() + y_pred.tolist()
# print(X)
# print(Y)
# print(Z)

####визуализация прогноза

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

C=plt.tricontour(X,Y,Z,colors='black',linewidths=1)
plt.tricontourf(X,Y,Z)
plt.clabel(C, inline=1, fontsize=10)
plt.show()



# df_test = pd.read_csv('Empty_part.csv')
# df_train = pd.read_csv('Training_wells.csv')
# #df_train = pd.read_csv('train.csv')
# # Загрузка данных
#
# df_test = df_test.drop(['Well'], axis=1)
# df_train = df_train.drop(['Well'], axis=1)
# print(df_train)
# print(df_test)
#
# df = df_train
# x = df['X']
# y = df['Y']
#
# fig, ax = plt.subplots()
#
# ax.scatter(x, y,c = 'red')    #  цвет точек
#
# df = df_test
# x = df['X']
# y = df['Y']
#
# ax.scatter(x, y, c = 'green')    #  цвет точек
# ax.set_facecolor('black')
#
# fig.set_figwidth(10)     #  ширина и
# fig.set_figheight(10)    #  высота "Figure"
#
# plt.show()
#
#
#
#
# df = pd.concat([df_train, df_test])#объединяем фреймы
# df['NTG'].fillna(0.0, inplace=True)#зануляем пропуски для построение поверхности
# print(df)
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# X = df['X']
# Y = df['Y']
# Z = df['NTG']
#
# C=plt.tricontour(X,Y,Z,colors='black')
# plt.tricontourf(X,Y,Z)
# plt.clabel(C, inline=1, fontsize=10)
# plt.show()
#
# fig = plt.figure()
# ax = Axes3D(fig)
# surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.show()
#
#
# dx = 0.01; dy = 0.01
# x = np.arange(-2.0,2.0,dx)
# y = np.arange(-2.0,2.0,dy)
# X,Y = np.meshgrid(x,y)
# print(X)
# def f(x,y):
#     return (1 - y**5 + x**5)*np.exp(-x**2-y**2)
#
# C = plt.contour(X,Y,f(X,Y),8,colors='black')
# plt.contourf(X,Y,f(X,Y),8)
# plt.clabel(C, inline=1, fontsize=10)
# plt.show()
#
# df = df_train
# x = df['X']
# y = df['Y']
# z =df['NTG']
# print(z)
# x,y,z = np.meshgrid(x,y,z)
# fig = plt.figure()
# ax = Axes3D(fig)
#
# ax.plot_surface(x,y,z,rstride=1, cstride=1,cmap=plt.cm.hot)
#
# plt.show()

# arrayColumBin = ['Tectonic regime',
#                  'Hydrocarbon type',
#                  'Reservoir status',
#                  'Structural setting',
#                  'Period',
#                  'Lithology']
#
# res_test = pd.concat([df_test,df_train_x ], ignore_index=True)
# res_train_x = pd.concat([df_test,df_train_x ], ignore_index=True)
#
#
# #проверка уникальных элементов для выборки
# for i in arrayColumBin:
#     #print(res_test[i].unique())
#     #print(res_train_x[i].unique())
#     # print(df_test['Tectonic regime'].unique())
#     if len(res_test[i].unique().tolist()) == len(res_train_x[i].unique().tolist()):
#         print('true')
#     else:
#         print('false')
#
# print(len(df_test))
# print(len(df_train_x))
#
#
# res_train_x = Binarizer(res_train_x, arrayColumBin)
# res_test= Binarizer(res_test, arrayColumBin)
# #тренировочные данные находяся в конце разделим их
# res_train_x = res_train_x[len(df_test):]
#
# res_test = res_test[:len(df_test)]
#
# res_train_x.to_csv('BinTrain.csv')
# res_test.to_csv('BinTest.csv')
#
#
# #knn
# knn = KNeighborsClassifier()
# knn.fit(res_train_x, df_train_y)
# print(knn.predict(res_test))
#
# predicted = knn.predict(res_test)
# predicted = pd.DataFrame(predicted)
# predicted.to_csv('OutputKNNTestAnswer.csv')
#
#
# ###деревья
# from sklearn.tree import DecisionTreeClassifier
#
# model = DecisionTreeClassifier()
# model = model.fit(res_train_x, df_train_y)
# print(model)
# # make predictions
# expected = df_test
# predicted = model.predict(res_test)
# predicted = pd.DataFrame(predicted)
#
# predicted.to_csv('OutputDTCTestAnswer.csv')
#
# #print(metrics.classification_report(expected, predicted))
# #print(metrics.confusion_matrix(expected, predicted))
#
#
# #df_test['abc']= knn.predict(res_test)
# #print (df_test)
# #df_test.to_csv('OutputTest.csv')
#
# ###проверка на тренировочных данных
# v2_train_df=shuffle(df_train)
# #v2_train_df=df_train
# numberV =270
# v2_train_df= Binarizer(v2_train_df, arrayColumBin)
#
#
# v2_test_df = v2_train_df[numberV:]
# v2_train_df = v2_train_df[:numberV]
#
#
# v2_train_df_y = v2_train_df['Onshore/Offshore']
# v2_test_df_y = v2_test_df['Onshore/Offshore']
#
# v2_train_df_x = v2_train_df.drop(['Onshore/Offshore'], axis=1)
# v2_test_df_x =  v2_test_df.drop(['Onshore/Offshore'], axis=1)
#
# v2_test_df = v2_test_df.values
# v2_train_df = v2_train_df.values
# v2_train_df_y = v2_train_df_y.values
# v2_test_df_y = v2_test_df_y.values
# v2_train_df_x.info()
# print("жопа")
# print(v2_train_df_y)
# print("жопа")
# print(v2_test_df_y)
# print("жопа")
# print(v2_train_df_x)
# print("жопа")
# print(v2_test_df_x)
# #####проверка точности кнн
# knn = KNeighborsClassifier(n_neighbors=4,weights='distance')
# knn.fit(v2_train_df_x, v2_train_df_y)
# print(knn.predict(v2_test_df_x))
# print(knn)
#
# expected = v2_test_df_y
# predicted = knn.predict(v2_test_df_x)
# print(predicted)
#
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# #df_train['res']= predicted
# #df_train.to_csv('OutputKNNTrain.csv')
# predicted = {0: predicted}
# test = pd.DataFrame(predicted)
#
# # print(test)
# #v2_train_df_x = pd.concat([v2_train_df_x, v2_test_df_x], ignore_index=True)
# #v2_train_df_y = pd.concat([v2_train_df_y, test], ignore_index=True)
#
# # мтеод логической регрессии
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# model.fit(v2_train_df_x, v2_train_df_y)
# print(model)
# # make predictions
# expected = v2_test_df_y
# predicted = model.predict(v2_test_df_x)
# print(predicted)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
#
# #метод наивный байес
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(v2_train_df_x, v2_train_df_y)
# print(model)
# # make predictions
# expected = v2_test_df_y
# predicted = model.predict(v2_test_df_x)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
#
# #метод деревья решений
# from sklearn.tree import DecisionTreeClassifier
# # fit a CART model to the data
#
# model = DecisionTreeClassifier()
# model= model.fit(v2_train_df_x, v2_train_df_y)
# print(model)
# # make predictions
# expected = v2_test_df_y
# predicted = model.predict(v2_test_df_x)
#
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
#
# dot_data = tree.export_graphviz(model, out_file='graph'+str(i),
# filled=True, rounded=True,special_characters=True)
# graph = graphviz.Source(dot_data)
#
#
#
#
# ##метод  опорных векторов
# from sklearn.svm import SVC
# # fit a SVM model to the data
# model = SVC()
# model.fit(v2_train_df_x, v2_train_df_y)
# print(model)
# # make predictions
# expected = v2_test_df_y
#
# predicted = model.predict(v2_test_df_x)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
