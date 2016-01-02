function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);


M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);
thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;

grad = [thetagrad(:)];
end

