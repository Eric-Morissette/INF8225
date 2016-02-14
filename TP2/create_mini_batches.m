function [ X_Batch, Y_Batch ] = create_mini_batches( X, Y, batchSize )
    size_X_2 = size( X, 2 );
    indexes = randperm( size_X_2 );

    shuffled_X = X( :, indexes );
    shuffled_Y = Y( indexes, : );

    sizes       = ones( 1, floor( size_X_2 / batchSize ) ) * batchSize;
    sizes(end)  = mod( size_X_2, batchSize ) + batchSize;

    X_Batch = mat2cell( shuffled_X, size( X, 1), sizes);
    Y_Batch = mat2cell( shuffled_Y, sizes, size( Y, 2));
end

