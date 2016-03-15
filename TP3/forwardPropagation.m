function [ a, w, f ] = forwardPropagation( X, Theta )
    sizeTheta = size(Theta, 2);

    % Create and fill w and a
    w = cell(1, size(Theta, 2));
    a = cell(1, size(Theta, 2));
	w{1} = [X' ; ones(1, size(X, 1))];
    for j = 1:size(Theta, 2)-1
        a{j} = Theta{j} * w{j};
        w{j+1} = [ ((a{j} >= 0) + 0.1 * (a{j} < 0)) .* a{j} ; ones(1, size(a{j}, 2))];
    end
    a{sizeTheta} = Theta{sizeTheta} * w{sizeTheta};

    %Use a mix of log and exp to minimise the impact/possibility of NaN due to over/under flow
    % http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
    % http://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
    maxA = max(a{sizeTheta}, [], 1);
    maxValue = repmat(maxA, size(a{sizeTheta}, 1), 1);
    AjminusMax = a{sizeTheta} - maxValue;
    f = exp( AjminusMax - repmat( log( sum( exp( AjminusMax) ) ), size(a{sizeTheta}, 1), 1) );
end