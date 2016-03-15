function [ precision ] = computePrecision( X, Y, Theta )
    [~, ~, f] = forwardPropagation(X, Theta);
    [~, predictions] = max(f);
    precision = sum( sum( predictions' == (Y' * [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]) ) ) / size(Y, 2);
end