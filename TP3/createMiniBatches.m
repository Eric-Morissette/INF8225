function [ X_Batch, Y_Batch ] = createMiniBatches( X, Y, batchSize )
    indices = randperm(size(X, 1));

    shuffled_X = X(indices, :);
    shuffled_Y = Y(:, indices);

    sizes = batchSize * ones(1, floor(size(X, 1) / batchSize));
    sizes(end) = batchSize + mod(size(X, 1), batchSize);

    X_Batch = mat2cell(shuffled_X, sizes, size(X, 2));
    Y_Batch = mat2cell(shuffled_Y, size(Y, 1), sizes);
end