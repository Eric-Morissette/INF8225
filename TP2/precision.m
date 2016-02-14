function [ precision ] = precision( X, Y, theta )
    p = exp( eye(4) * theta * X) ./ ( repmat( sum( exp( eye(4) * theta * X) ), 4, 1) );
    [M, predictions] = max(p);
    corrects = sum( predictions' == ( Y * [1; 2; 3; 4] ) );
    precision = corrects / size(X,2);
end

