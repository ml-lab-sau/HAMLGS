function W = optimize_W(X, Y, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, iterations, threshold, eta)
    % Initialize variables
    [n, d] = size(X);
    [~, c] = size(Y);
   % [~, c] = size(E);
    U1= zeros(n);
    E=ones(n,c);
    sigma=1;
    L=computeLaplacian(X,sigma);
    large_value = 1e10;  
    % Initialization
    W = rand(d, c);
    D = rand(n, d);
    N = X - D;
    %F = rand(n, c);
    F = ones(n, c);
for i = 1:n
    if any(Y(i, :) ~= 0)
        U1(i,i) = large_value;
    else
        U1(i,i) = 0;
    end
end
    % Main loop
    for t = 1:iterations
        % Update W
        gradient_W = - D' * (max(0, abs(E - F .* (D * W))) .* (-F)) + lambda1 * W + 2 * lambda6 * D' * D * W;
        W = W - eta * gradient_W;

        % Update D
        gradient_D = (max(0, abs(E - F .* (D * W))) .* (-F)) * W' - lambda4 * (X - D - N) + 2 * lambda6 * D * (W * W');
        D = D - eta * gradient_D;

        % Update N
        gradient_N = lambda2 * proximal_operator(N, lambda2 / eta) - lambda5 * (X - D - N);
        N = N - eta * gradient_N;

        % Update F
        gradient_F =-D*W.*(max(0, abs(E - F .* (D * W))))+lambda3 * L * F + lambda5 * U1 * (F - Y);
        F = F - eta * gradient_F;

        % Check convergence
        if norm(gradient_W, 'fro') / norm(W, 'fro') < threshold
            break;
        end
    end

    % Proximal operator for nuclear norm
    function result = proximal_operator(M, tau)
        if any(isnan(M(:))) || any(isinf(M(:)))
        % Handle NaN or Inf values
        % Replace or remove problematic values based on your specific scenario
        % For example, you can replace them with zeros
        M(isnan(M) | isinf(M)) = 0;
    end
        [U, S, V] = svd(M);
        S = max(S - tau, 0);
        result = U * S * V';
    end
end
function L = computeLaplacian(X, sigma)
    % Step 1: Construct the similarity matrix S
    n = size(X, 1);
    S = zeros(n, n);
    for i = 1:n
        for j = 1:n
            S(i, j) = exp(-norm(X(i, :) - X(j, :))^2 / (2 * sigma^2));
        end
    end
    
    % Step 1a: Normalize the similarity matrix S
    S = S ./ sum(S, 2);

    % Step 2: Compute the diagonal matrix Lambda
    Lambda = diag(sum(S, 2));

    % Step 3: Calculate the Laplacian matrix L
    L = Lambda - S;
end

