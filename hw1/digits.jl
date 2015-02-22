using SVM
using MAT

# matwrite("C://Users//Austin//Documents//GitHub//cs189//hw1//matfile.mat", {
#     "myvar1" => 0,
#     "myvar2" => 1 })
vars = matread("C:/Users/Austin/Documents/GitHub/cs189/hw1/data/digit-dataset/train.mat")
images = vars["train_images"]
labels = vars["train_labels"]

# image1 = images[:, :, 1]
# image1x = vec(image1)
# image1y = reshape(image1, (784))
# X = hcat(vec(images[:, :, i]) for i=1:6)
# p, n = size(X)

n = 60000

XX = Array(Float64, (784, n))
for i = 1:n
  XX[:, i] = vec(images[:, :, i])
end
p, n = size(XX)

Y = [label == 0 ? 1.0 : -1.0 for label in labels[1:n]]

train = randbool(n)
model = svm(XX[:, train], Y[train], T = 10)
accuracy = countnz(predict(model, XX[:,~train]) .== Y[~train])/countnz(~train)

#######################
using RDatasets

# We'll learn to separate setosa from other species
iris = dataset("datasets", "iris")
iris[:, 1:4]
# SVM format expects observations in columns and features in rows
X = array(iris[:, 1:4])'
p, n = size(X)

# SVM format expects positive and negative examples to +1/-1
Y = [species == "setosa" ? 1.0 : -1.0 for species in iris[:Species]]

# Select a subset of the data for training, test on the rest.
train = randbool(n)

# We'll fit a model with all of the default parameters
model = svm(X[:,train], Y[train])

# And now evaluate that model on the testset
accuracy = countnz(predict(model, X[:,~train]) .== Y[~train])/countnz(~train)

