function [ X_Learning, X_Validation, X_Test, Y_Learning, Y_Validation, Y_Test ] = create_train_valid_test_splits( X, Y )
    distributions = [0.7 0.15 0.15];
    distributions = floor(distributions * size(X,2));
    indexes = randperm(size(X,2));

    e = cumsum(distributions);
    b = e - distributions + ones(1, size(distributions, 2));

    X_Learning      = X(:, indexes(b(1):e(1)));
    X_Validation    = X(:, indexes(b(2):e(2)));
    X_Test          = X(:, indexes(b(3):e(3)));

    Y_Learning      = Y(indexes(b(1):e(1)), :);
    Y_Validation    = Y(indexes(b(2):e(2)), :);
    Y_Test          = Y(indexes(b(3):e(3)), :);
end