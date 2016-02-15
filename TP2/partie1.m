% Clear console and workspace
clc;
clear;

% Load the file
load 20news_w100;

% Initialize dimensions
n = 4;
m = size(newsgroups, 2);
o = ones(1, m);
i = 1:m;
j = newsgroups;
Y = sparse(i, j, o, m, n);

theta           = rand(4, 101) - 0.5;
X               = documents;
X               = [X; ones(1, 16242)];
rate_Learning   = 0.0005;

% Split the data into Learning, Validation and Test data
[X_Learning, X_Validation, X_Test, Y_Learning, Y_Validation, Y_Test] = create_train_valid_test_splits(X, Y);

%%%%%%%%%%%%%%%%%%%%%
%       Batch       %
%%%%%%%%%%%%%%%%%%%%%
batch = Y_Learning' * X_Learning';

itr = 0;
logLikelihood           = -realmax;
precision_Learning      = [];
precision_Validation    = [];
logLikelihood_Batch     = [];

loglikelihoodDifference = 2;
while abs(loglikelihoodDifference) > 1
	% Compute LogLikelihood
    tempLogLikelihood       = sum((sum( ( ( Y_Validation * theta) .* X_Validation')' )) - (log(sum( exp( eye(4) * theta * X_Validation) ))));
    loglikelihoodDifference = tempLogLikelihood - logLikelihood;
    logLikelihood           = tempLogLikelihood;
    logLikelihood_Batch     = [logLikelihood_Batch logLikelihood];

    precision_Learning      = [precision_Learning precision(X_Learning, Y_Learning, theta)];
    precision_Validation    = [precision_Validation precision(X_Validation, Y_Validation, theta)];

    expected    = (exp( eye(4) * theta * X_Learning) ./ (repmat( sum( exp( eye(4) * theta * X_Learning) ), 4, 1))) * X_Learning';
    gradient    = expected - batch;
    theta       = theta - rate_Learning * gradient;

    itr = itr + 1;
end
x1 = 1:itr;

fprintf('Precision of the batch descent with test data = %f\n', precision(X_Test, Y_Test, theta));


%%%%%%%%%%%%%%%%%%%%%
%     Mini-Batch    %
%%%%%%%%%%%%%%%%%%%%%
theta                           = rand(4, 101) - 0.5;
miniBatchSize                   = 568;
alpha                           = 0.6;
thetaDifference                 = zeros(4, 101);
itr                             = 0;
logLikelihood                   = -realmax;
logLikelihood_MiniBatch         = [];
precision_Validation_MiniBatch  = [];
precision_Learning_MiniBatch    = [];
delta                           = 1;

converged = false;
while ~converged
    itr = itr + 1;
    [X_Batch, Y_Batch] = create_mini_batches(X_Learning, Y_Learning, miniBatchSize);

    rate_Learning = 2 / itr;
    for i = 1:size( X_Batch, 2)
        tempLogLikelihood       = sum( ( sum( ( ( Y_Validation * theta ) .* X_Validation' )' ) ) - log( sum( exp( eye(4) * theta * X_Validation ) ) ) );
        delta                   = tempLogLikelihood - logLikelihood;
        logLikelihood           = tempLogLikelihood;
        logLikelihood_MiniBatch = [logLikelihood_MiniBatch logLikelihood];

        expected        = ( ( exp( eye(4) * theta * X_Batch{:,i} ) ) ./ ( repmat( sum( exp( eye(4) * theta * X_Batch{:,i} ) ), 4, 1 ) ) ) * X_Batch{:,i}';
        batch           = Y_Batch{i, :}' * X_Batch{:, i}';
        gradient        = (expected - batch) ./ miniBatchSize;
        thetaDifference = (alpha * thetaDifference) - (rate_Learning * gradient);
        theta           = theta + thetaDifference;

        precision_Validation_MiniBatch = [precision_Validation_MiniBatch precision(X_Validation, Y_Validation, theta)];

        if abs(delta) < 0.0001
            converged = true;
            break;
        end
    end

    precision_Learning_MiniBatch = [precision_Learning_MiniBatch precision(X_Learning, Y_Learning, theta)];
end
x2 = ( 1:size( logLikelihood_MiniBatch, 2 ) ) / size( X_Batch, 2 );
x3 = 1:itr;

fprintf('Precision of the mini-batch descent with test data = %f\n', precision(X_Test, Y_Test, theta));

figure();
plot(x1, logLikelihood_Batch, x2, logLikelihood_MiniBatch);
title('LogLikelihood in relation with the iteration');
xlabel('Iteration');
ylabel('LogLikelihood');
legend('LogLikelihood (Batch)', 'LogLikelihood (Mini-Batch)');

figure();
plot(x1, precision_Learning, x1, precision_Validation, x2, precision_Validation_MiniBatch, x3, precision_Learning_MiniBatch);
title('Learning Curves');
xlabel('Iteration');
ylabel('Precision');
legend('Learning Data (Batch)','Validation Data (Batch)', 'Validation Data (Mini-Batch)', 'Learning Data (Mini-Batch)');