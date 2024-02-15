function adj_matrices = heat_kernel_graphs(X, t_values)
    % Function to calculate pairwise Euclidean distances
    euclidean_distances = @(X) sqrt(sum((X(:, 1:end-1) - X(:, 2:end)).^2, 2));

    % Calculate the number of instances
    n = size(X, 1);

    % Initialize cell array to store adjacency matrices
    adj_matrices = cell(1, length(t_values));

    % Calculate the adjacency matrices for each value of t
    for i = 1:length(t_values)
        t = t_values(i);

        % Calculate the pairwise Euclidean distances
        distances = squareform(pdist(X));

        % Calculate the heat kernel matrix A
        A = exp(-distances.^2 / (2 * t * mean(distances(:))^2));

        % Ensure diagonal elements are 0 (optional, depends on application)
        A = A - diag(diag(A));

        % Store the adjacency matrix in the cell array
        adj_matrices{i} = A;
    end
end
