function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)


theta = 0.005 * randn(numClasses * inputSize, 1);


addpath minFunc/
options.Method = 'lbfgs'; 


[softmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, numClasses, inputSize, lambda, inputData, labels),  theta, options);
% [f,g] = softmax_regression_vec(theta, X, y)
                          


softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          
