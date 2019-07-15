#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
import ipywidgets as widgets
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import interact, interactive, fixed, interact_manual, IntSlider


# Data used is from UCI's Machine Learning Repository. The specific set can be found at https://archive.ics.uci.edu/ml/datasets/seeds.
# This dataset, per the UCI website, contains "measurements of geometrical properties of kernels to belonging to three different varieties of wheat." My goal in this project is to test classification techniques using Guassian distributions to see if the algorithm can accurately predict class 1 - Kama, class 2 - Rosa, or class 3 - Canadian. There are 70 examples of each class in the given data set. 

# In[2]:


#loading text file and examining the shape to be sure that the website description 
#of 210 instances with 7 attributes each, and see at which index the class feature is found.
data = np.loadtxt('seeds_dataset.txt')
print(data.shape)
print(data[1])


# According to the aforementioned website the indices are labelled as follows:
# 0 - area A of the kernel
# 1 - perimiter P
# 2 - compactness where C = 4(pi)A/P^2
# 3 - length of the kernel
# 4 - width of the kernel
# 5 - asymmetry coefficient
# 6 - length of kernel groove
# 7 - Class (1, 2, 3)
# 
# FOR PURPOSES OF THE MACHINE LEARNING STUDY, I WILL NOT FOCUS ON THE SPECIFICATIONS OF WHAT EACH OF THESE FEATURES MEAN, RATHER I WILL TAKE THEM AS GIVEN AND USE THEM TO TEST CLASSIFICATION ALGORITHMS.

# In[3]:


#Let's put the feature names into a variable for future use
featureNames = ['area', 'perimiter', 'compactness', 'lengthKernel', 'widthKernel',
               'asymmetryCoefficient', 'lengthGroove']


# In[4]:


#Next let's split the data into training and test sets using random permutations
#in case the dataset is in a strict order (which it is in this particular case)
np.random.seed(0)
perm = np.random.permutation(len(data)) #because there are 210 seeds tested
trainx = data[perm[0:150], 0:7] #will use 5/7 of the data as training
trainy = data[perm[0:150], 7]
testx = data[perm[150:210], 0:7] #2/7 for testing
testy = data[perm[150:210], 7]


# In[5]:


#Let's make sure the data is appropriate by counting the number of training points from each class
sum(trainy == 1), sum(trainy ==  2), sum(trainy == 3)


# Good! it looks like the data was split nicely.
# I will start by looking at the distribution of each single feature with it's Gaussian fit.
# I am using a slider so that inside the code we can easily look at the distributions of each of the three classes over their 7 features. 

# In[6]:


@interact_manual(feature = IntSlider(0,1,6), label = IntSlider(1,1,3)) 
def density_plot(feature, label):
        plt.hist(trainx[trainy == label, feature], normed = True)
        mu = np.mean(trainx[trainy == label, feature])
        var = np.var(trainx[trainy == label, feature])
        std = np.sqrt(var)
        x_axis = np.linspace(mu-3*std, mu+3*std, 500)
        plt.plot(x_axis, norm.pdf(x_axis, mu, std), 'r', lw = 2)
        plt.title("Seed" + str(label))
        plt.xlabel(featureNames[feature], fontsize = 10, color = 'red')
        plt.ylabel('Density', fontsize = 10, color = 'red')
        plt.show()


# Now let's compare the Gaussian of each class in each specific feature. First, we need to define some function that will fit the model to each class using a single (yet adjustable) feature. 

# In[7]:


def fit_model(x,y,feature):
    k = 3 #we have 3 classes
    mu = np.zeros(k+1)
    var = np.zeros(k+1)
    pi = np.zeros(k+1) #class weights
    for label in range(1,k+1):
        indices = (y == label)
        mu[label] = np.mean(x[indices, feature])
        var[label] = np.var(x[indices, feature])
        pi[label] = float(sum(indices))/float(len(y))
    return mu, var, pi


# Let's print 2 examples of class weights just to make sure the function appears to work appropriately.

# In[8]:


mu,var,pi = fit_model(trainx, trainy, 2)
print(mu)
print(var)
print(pi)


# Good! we have a function to calculate the weights for each class using a given feature, and it can be used in side the function to create a comparative graph of each class over each feature.

# In[9]:


@interact_manual(feature = IntSlider(0,0,6))
def show_densities(feature):
    mu, var, pi = fit_model(trainx, trainy, feature)
    colors = ['g', 'r', 'b']
    for label in range(1,4):
        m = mu[label]
        s = np.sqrt(var[label])
        x_axis = np.linspace(m-3*s, m+3*s, 1000)
        plt.plot(x_axis, norm.pdf(x_axis, m, s), colors[label-1],
                label = 'Seed' + str(label))
        plt.xlabel(featureNames[feature], fontsize = 10, color = 'green')
        plt.ylabel('Density', fontsize = 10, color = 'green')
        plt.legend()
    plt.show()
    print(mu)


# By running through each of the feature density comparisons above, one cane see that there will more than likely be a high amount of test error, so let's write a function to calculate that error.

# In[10]:


@interact(feature = IntSlider(0,0,6))
def test_model(feature):
    mu, var, pi = fit_model(trainx, trainy, feature)
    k = 3
    n_test = len(testy)
    score = np.zeros((n_test, k+1))
    for i in range(0, n_test):
        for label in range(1, k+1):
            score[i, label] = np.log(pi[label]) +             norm.logpdf(testx[i, feature], mu[label], 
            np.sqrt(var[label]))
    pred = np.argmax(score[:,1:4], axis = 1) + 1
    errors = np.sum(pred != testy)
    print("Errors made using" + ' ' + featureNames[feature] + ": " + str(errors) + '/' + str(n_test))


# The first feature was our only reasonably successful predictor feature, returning an error of 3/20 on the test set. All other features returned an error of over 50%. Perhaps for this dataset using a univariate model is not the way to go. Let's try it with a bivariate Guassian and see what happens...

# In[11]:


#Creating a helper function to fit the Gaussian for more than one feature
def fit_model(x, features):
    mu = np.mean (x[:,features], axis = 0)
    covariance = np.cov(x[:, features], rowvar = 0, bias = 1)
    return mu, covariance


# In[12]:


#this will give us the mean of the two features and a covariance matrix
label = 1 
mu, covariance = fit_model(trainx[trainy == label, :], [0,1])
print(mu)
print(covariance)


# In[13]:


#creating a helper function to find the range for which the values of selected features lie
def findRange(x):
    lower = min(x)
    upper = max(x)
    width = upper - lower
    lower = lower - 0.2*width
    upper = upper + 0.2*width
    return lower, upper


# In[14]:


print(findRange(trainx[trainy == 1, 1])) #test example to make sure it works


# Now that that's finished, a function to plot contour lines on a grid for the two-dimensional Gaussian is necessary. Let's establish a function for that as well as a function to actually plot it all together. 

# In[15]:


def contours(mu, cov, x1grid, x2grid, col):
    rv = multivariate_normal(mean = mu, cov = cov)
    z = np.zeros((len(x1grid), len(x2grid)))
    for i in range(0, len(x1grid)):
        for j in range(0, len(x2grid)):
            z[j,i] = rv.logpdf([x1grid[i], x2grid[j]])
    sign, logdet = np.linalg.slogdet(cov)
    normalizer = -0.5 * (2*np.log(6.28) + sign * logdet)
    for offset in range(1,4):
        plt.contour(x1grid, x2grid, z, 
                   levels = [normalizer - offset], colors = col,
                   linewidths = 2.0, linestyles = 'solid')


# In[16]:


@interact_manual(f1 = IntSlider(0,0,6, 1), f2 = IntSlider(0,0,6,1),
                 label = IntSlider(1,1,3,1))
def bivariatePlot(f1, f2, label):
    if f1 == f2:
        print("Please choose two features that are different.")
        return 
    x1_lower, x1_upper = findRange(trainx[trainy == label, f1])
    x2_lower, x2_upper = findRange(trainx[trainy == label, f2])
    plt.xlim(x1_lower, x1_upper)
    plt.ylim(x2_lower, x2_upper)
    plt.plot(trainx[trainy == label, f1], trainx[trainy == label, f2], 'ro')
    res = 300
    x1grid = np.linspace(x1_lower, x1_upper, res)
    x2grid = np.linspace(x2_lower, x2_upper, res)
    mu, cov = fit_model(trainx[trainy == label, :], [f1,f2])
    contours(mu, cov, x1grid, x2grid, 'k')
    plt.xlabel(featureNames[f1], fontsize = 10, color = 'g')
    plt.ylabel(featureNames[f2], fontsize = 10, color = 'g')
    plt.title('Seed' + str(label), fontsize = 10, color = 'g')
    plt.show()
    


# This is great, the plots above show us some strong correlations between features. As before, let's now plot the three Gaussians on the same graph, this time in relation to two features each.

# In[17]:


def fit_model_bivariate(x, y, features):
    k = 3
    d = len(features)
    mu = np.zeros((k+1, d))
    covar = np.zeros((k+1, d, d))
    pi = np.zeros(k+1)
    for label in range(1, k+1):
        indices = (y == label)
        mu[label,:], covar[label,:,:] = fit_model(x[indices,:], features)
        pi[label] = float(sum(indices))/float(len(y))
    return mu, covar, pi


# In[18]:


@interact_manual(f1 = IntSlider(0, 0, 6, 1), f2 = IntSlider(0,0,6,1))
def threePlot(f1,f2):
    if f1 == f2:
        print("Please choose features that are different from each other")
        return
    x1_lower, x1_upper = findRange(trainx[:, f1])
    x2_lower, x2_upper = findRange(trainx[:, f2])
    plt.xlim(x1_lower, x1_upper)
    plt.ylim(x2_lower, x2_upper)
    colors = ['g', 'b', 'r']
    for label in range(1,4):
        plt.plot(trainx[trainy == label, f1], trainx[trainy == label, f2], marker = 'o', ls = 'None', c = colors[label-1])
    res = 400
    x1grid = np.linspace(x1_lower, x1_upper, res)
    x2grid = np.linspace(x2_lower, x2_upper, res)
    mu, covar, pi = fit_model_bivariate(trainx, trainy, [f1,f2])
    for label in range(1, 4):
        gmean = mu[label,:]
        gcov = covar[label,:,:]
        contours(gmean, gcov, x1grid, x2grid, colors[label-1])
    plt.xlabel(featureNames[f1], fontsize = 10, color = 'black')
    plt.ylabel(featureNames[f2], fontsize = 10, color = 'black')
    plt.title("Wheat Seeds Comparison", fontsize = 10, color = 'black')
    plt.show()


# In[19]:


@interact(f1 = IntSlider(0,0,6,1), f2 = IntSlider(0,0,6,1))
def test(f1,f2):
    if f1 == f2:
        print("Please choose features that are different from each other")
        return
    features = [f1,f2]
    mu, covar, pi = fit_model_bivariate(trainx, trainy, features)
    k = 3
    nt = len(testy)
    score = np.zeros((nt, k+1))
    for i in range(0,nt):
        for label in range(1, k+1):
            score[i, label] = np.log(pi[label]) +            multivariate_normal.logpdf(testx[i,features], mean = 
            mu[label,:], cov = covar[label,:,:])
    preds = np.argmax(score[:,1:4], axis = 1) + 1
    errors = np.sum(preds != testy)
    print("Errors using features" + " " + str(f1) + " and " + str(f2) + ": " + str(errors) + '/' + str(nt))


# Scrolling through the errors made on the test set using the bivariate Gaussian shows a great improvement over the univariate test. The lowest amount of errors made was 0, a perfect score (using features 3 and 6), and the highest was 19 (using features 2 and 5). 

# In[ ]:





# In[ ]:




