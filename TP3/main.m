clc;
clear;

load mnist_all_v2;

X_Training      = logical(train_x);
X_Validation    = logical(valid_x);
X_Testing       = logical(test_x);

Y_Training      = sparse(1:50000, double(train_y) + ones(1, 50000), ones(1, 50000), 50000, 10)';
Y_Validation    = sparse(1:10000, double(valid_y) + ones(1, 10000), ones(1, 10000), 10000, 10)';
Y_Testing       = sparse(1:10000, double(test_y)  + ones(1, 10000), ones(1, 10000), 10000, 10)';

tic

layers = [784 387 10];
batchSize = 100;
learningRate = 0.004;
lambda1 = 0.0;
lambda2 = 1.5;
nbItr = 50;
createAndTrainNN(X_Training, Y_Training, X_Validation, Y_Validation, X_Testing, Y_Testing, layers, batchSize, learningRate, lambda1, lambda2, nbItr);

toc

