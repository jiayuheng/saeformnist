%this code is based on Stanford University DeepLearning tutorial.
clc
clear

tic
inputSize = 28 * 28;
numClasses = 10;
% set the structure of  of nn
hiddenSizeL1 = 100;    % Layer 1 Hidden Size
hiddenSizeL2 = 64;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average sparse rato.

lambda = 3e-3;         % regularization parameter       
beta = 3;              % sparsity penalty term       


%%  Load data

addpath(genpath('./minFunc_2012/minFunc'));


load MNIST
%% debug
% trainData=trainData(:,1:1000);
% trainLabels=trainLabels(1:1000);

trainLabels(trainLabels == 0) = 10; 


%% train first sae or can replace this by rbm of something
sae1Theta = parainit(hiddenSizeL1, inputSize);

% optimization algorithm
options = struct;
options.Method = 'lbfgs';
options.maxIter = 200;
% options.display = 'on';
[sae1OptTheta, cost] =  minFunc(@(p)saecost(p,inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),sae1Theta,options);% minfunc is a kind of newtown method




%% train the second sae 
[sae1Features] = saeoutput(sae1OptTheta, hiddenSizeL1, inputSize, trainData);%get feature by the first layer

sae2Theta = parainit(hiddenSizeL2, hiddenSizeL1);


[sae2OptTheta, cost] =  minFunc(@(p)saecost(p,hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);





%% train softmax

[sae2Features] = saeoutput(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features);


saeSoftmaxTheta = 0.5 * randn(hiddenSizeL2 * numClasses, 1);



softmaxLambda = 1e-4;
numClasses = 10;
softoptions = struct;
softoptions.maxIter = 400;
softmaxModel = softmaxTrain(hiddenSizeL2,numClasses,softmaxLambda,sae2Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);



%% fineturn the para by bp


stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);

stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1),hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);%code by andrew ng

stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];


[stackedAEOptTheta, cost] =  minFunc(@(p)msaecost(p,inputSize,hiddenSizeL2,numClasses, netconfig,lambda, trainData, trainLabels),stackedAETheta,options);



  optStack = params2stack(stackedAEOptTheta(hiddenSizeL2*numClasses+1:end), netconfig);% ng
  W11 = optStack{1}.w;
  W12 = optStack{2}.w;



%% test

testLabels(testLabels == 0) = 10; % Remap 0 to 10

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2,numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);
toc

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, trainData);
acc = mean(trainLabels(:) == pred(:));

