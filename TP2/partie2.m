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

X = documents;
X = [ X; ones( 1, m ); randi( [0 1], 100, m ) ];

% Split the data into Learning, Validation and Test data
[ X_Learning, X_Validation, X_Test, Y_Learning, Y_Validation, Y_Test ] = create_train_valid_test_splits( X, Y );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Mini-Batch without Regularization  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta           = rand(4, 201) - 0.5;
miniBatchSize   = 568;
alpha           = 0.6;
thetaDifference = zeros(4, 201);
itr             = 0;
logLikelihood   = -realmax;

converged = false;
while ~converged
    itr = itr + 1;
    [ X_Batch, Y_Batch ] = create_mini_batches( X_Learning, Y_Learning, miniBatchSize );

    learning_Rate = 2 / itr;
    for i = 1:size( X_Batch, 2)
        tempLogLikelihood   = sum( ( sum( ( ( Y_Validation * theta ) .* X_Validation' )' ) ) - log( sum( exp( eye(4) * theta * X_Validation ) ) ) );
        delta               = tempLogLikelihood - logLikelihood;
        logLikelihood       = tempLogLikelihood;

        expected        = ( ( exp( eye(4) * theta * X_Batch{:,i} ) ) ./ ( repmat( sum( exp( eye(4) * theta * X_Batch{ :, i } ) ), 4, 1 ) ) ) * X_Batch{ :, i }';
        batch           = Y_Batch{i, :}' * X_Batch{:, i}';
        gradient        = ( expected - batch ) ./ miniBatchSize;
        thetaDifference = ( alpha * thetaDifference ) - ( learning_Rate * gradient );
        theta           = theta + thetaDifference;

        if abs( delta ) < 0.0001
            converged = true;
            break;
        end
    end
end

fprintf('Precision of the mini-batch descent without regularization = %f\n', precision(X_Test, Y_Test, theta));

figure();
subplot( 2, 2, 1 );
histogram( abs( theta( 1, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 1)'; 'No Regularization' } );
xlabel( 'Weight');
ylabel( 'Occurrences');

subplot( 2, 2, 2 );
histogram( abs( theta(2, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 2)'; 'No Regularization' } );
xlabel( 'Weight');
ylabel( 'Occurrences');

subplot( 2, 2, 3 );
histogram( abs( theta( 3, 1:101 ) ), 20 );
title( {'Original Parameters Histogram (Y = 3)'; 'No Regularization' } );
xlabel( 'Weight' );
ylabel( 'Occurrences' );

subplot( 2, 2, 4 );
histogram( abs( theta( 4, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 4)'; 'No Regularization' } );
xlabel( 'Weight' );
ylabel( 'Occurrences' );


figure();
histogram( abs( theta( :, 102:end ) ), 20 );
title( { 'Random Parameters Histogram'; 'No Regularization' } );
xlabel( 'Weight ');
ylabel( 'Occurrences' );



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Mini-Batch with Regularization    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta           = rand(4, 201) - 0.5;
miniBatchSize   = 568;
alpha           = 0.6;
thetaDifference = zeros(4, 201);
lambda1         = 0.015;
lambda2         = 0.050;
itr             = 0;
logLikelihood   = -realmax;

converged = false;
while ~converged
    itr = itr + 1;
    [X_Batch, Y_Batch] = create_mini_batches(X_Learning, Y_Learning, miniBatchSize);

    learning_Rate = 2 / itr;
    for i = 1:size( X_Batch, 2)
        tempLogLikelihood   = sum( ( sum( ( ( Y_Validation * theta ) .* X_Validation' )' ) ) - log( sum( exp( eye(4) * theta * X_Validation ) ) ) );
        delta               = tempLogLikelihood - logLikelihood;
        logLikelihood       = tempLogLikelihood;

        expected        = ( ( exp( eye(4) * theta * X_Batch{:,i} ) ) ./ ( repmat( sum( exp( eye(4) * theta * X_Batch{:,i} ) ), 4, 1 ) ) ) * X_Batch{:,i}';
        batch           = Y_Batch{i, :}' * X_Batch{:, i}';
        gradient        = ( ( expected - batch ) ./ miniBatchSize ) + ( ( miniBatchSize / size( X_Learning, 2 ) ) * ((lambda1 * 2 * theta ) + ( lambda2 * ( ( ( theta > 0 ) + ( theta < 0 ) * -1 ) ) ) ) );
        thetaDifference = ( alpha * thetaDifference ) - ( learning_Rate * gradient );
        theta           = theta + thetaDifference;

        if abs( delta ) < 0.01
            converged = true;
            break;
        end
    end
end

fprintf('Precision of the mini-batch descent with regularization = %f\n', precision(X_Test, Y_Test, theta));

figure();
subplot( 2, 2, 1 );
histogram( abs( theta( 1, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 1)'; 'With Regularization' } );
xlabel( 'Weight');
ylabel( 'Occurrences');

subplot( 2, 2, 2 );
histogram( abs( theta(2, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 2)'; 'With Regularization' } );
xlabel( 'Weight');
ylabel( 'Occurrences');

subplot( 2, 2, 3 );
histogram( abs( theta( 3, 1:101 ) ), 20 );
title( {'Original Parameters Histogram (Y = 3)'; 'With Regularization' } );
xlabel( 'Weight' );
ylabel( 'Occurrences' );

subplot( 2, 2, 4 );
histogram( abs( theta( 4, 1:101 ) ), 20 );
title( { 'Original Parameters Histogram (Y = 4)'; 'With Regularization' } );
xlabel( 'Weight' );
ylabel( 'Occurrences' );


figure();
histogram( abs( theta( :, 102:end ) ), 20 );
title( { 'Random Parameters Histogram'; 'With Regularization' } );
xlabel( 'Weight ');
ylabel( 'Occurrences' );


























