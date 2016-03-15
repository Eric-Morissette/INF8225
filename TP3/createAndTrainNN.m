function [ Theta ] = createAndTrainNN( X_Training, Y_Training, X_Validation, Y_Validation, X_Testing, Y_Testing, layerSizes, batchSize, learningRate, lambda1, lambda2, maxItr )
    hiddenLayers = size(layerSizes, 2) - 2;

    % Initialize Theta with random parameters
    Theta = cell(1, size(layerSizes, 2) - 1);
    for i = 2:size(layerSizes, 2)
        Theta{i-1} = (rand( layerSizes(i), layerSizes(i-1) + 1) - 0.5) / sqrt(1 + layerSizes(i - 1));
    end

    % Loop-specific variables
    itr                     = 0;
    maxPrecision            = 0;
    precision_Training      = zeros(maxItr);
    precision_Validation    = zeros(maxItr);
    delta                   = cell(hiddenLayers + 1);
    for k = 1:maxItr
        itr = itr + 1;
        [X_Batch, Y_Batch] = createMiniBatches(X_Training, Y_Training, batchSize);
        for i = 1:size(X_Batch, 1)
            %FWD Prop
            [a, h, f] = forwardPropagation(X_Batch{i}, Theta);

            %BWD Prop
            delta{hiddenLayers + 1} = -(Y_Batch{i} - f);
            for j = hiddenLayers:-1:1
                delta{j} = ((a{j} >= 0) + 0.1 * (a{j} < 0)) .* (Theta{j+1}(:, 1:end-1)' * delta{j + 1});
            end

            %Regularization
            for j = 1:hiddenLayers + 1
                Theta{j} = (1 - learningRate * lambda2 * batchSize / size(X_Training, 1)) * Theta{j} - (learningRate / batchSize) * delta{j} * h{j}' - learningRate * lambda1 * batchSize / size(X_Training, 1) * sign(Theta{j});
            end
        end

        % Validation Set
        precision_Validation(k) = computePrecision(X_Validation, Y_Validation, Theta);
        if precision_Validation(k) > maxPrecision
            maxPrecision = precision_Validation(k);
            bestTheta = Theta;
        end

        % Training Set
        precision_Training(k) = computePrecision(X_Training, Y_Training, Theta);

        fprintf('Iteration  : %d\n', itr);
        fprintf('Max        : %f\n', maxPrecision);
        fprintf('Validation : %f\n', precision_Validation(k));
        fprintf('Training   : %f\n\n', precision_Training(k));
    end

    % Final Test
    fprintf('Final Test : %f\n', computePrecision(X_Testing, Y_Testing, bestTheta));

    % Plot Validation and Training
    figure();
    plot(1:maxItr, precision_Validation, 1:maxItr, precision_Training);
    legend('Validation', 'Training');
    title_ = sprintf('Precision with learning rate = %f', learningRate);
    title(title_);
end