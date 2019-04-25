import numpy as np  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def draw_vector(v0, v1, ax=None):

    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def pca():
    rng = np.random.RandomState(1)
    test_data = np.dot(rng.rand(2, 2), rng.randn(2, 100)).T

    test_data_flipped_y = np.array([test_data[:, 0], -test_data[:, 1]]).transpose()
    test_data = test_data_flipped_y

    test_data[:, 0] = np.interp(test_data[:, 0], (test_data[:, 0].min(), test_data[:, 0].max()), (-0.1, 0.1))
    test_data[:, 1] = np.interp(test_data[:, 1], (test_data[:, 1].min(), test_data[:, 1].max()), (-0.45, 0.45))

    plt.scatter(test_data[:, 0], test_data[:, 1], s=2)

    np.set_printoptions(suppress=True)

    feature_data = test_data
    pca = PCA(n_components=2)
    pca.fit(feature_data)
    print("Principal Components: ")
    print(pca.components_)
    print("Explained Variance: ")
    print(pca.explained_variance_)

    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 2 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')

    plt.ylim((-0.45, 0.45))
    plt.xlim((-0.1, 0.1))
    plt.show()

def pca_x_shaped():

    rng = np.random.RandomState(1)
    test_data = np.dot(rng.rand(2, 2), rng.randn(2, 100)).T

    test_data_flipped_y =  np.array([test_data[:, 0], -test_data[:, 1]]).transpose()
    test_data = np.concatenate((test_data, test_data_flipped_y), axis=0)

    test_data[:, 0] = np.interp(test_data[:, 0], (test_data[:, 0].min(), test_data[:, 0].max()), (-0.3, 0.3))
    test_data[:, 1] = np.interp(test_data[:, 1], (test_data[:, 1].min(), test_data[:, 1].max()), (-0.6, 0.6))

    plt.scatter(test_data[:, 0], test_data[:, 1], s=2)

    np.set_printoptions(suppress=True)

    feature_data = test_data
    pca = PCA(n_components=2)
    pca.fit(feature_data)
    print("Principal Components: ")
    print(pca.components_)
    print("Explained Variance: ")
    print(pca.explained_variance_)

    origin = [0], [0]
    plt.quiver(*origin, pca.components_[:, 0] * 7 * pca.explained_variance_[1], pca.components_[:, 1] * 7 * pca.explained_variance_[0], color=['r', 'b'], scale=1)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 2 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')

    plt.ylim((-0.61, 0.61))
    plt.xlim((-0.35, 0.35))
    plt.show()


#pca()
pca_x_shaped()
