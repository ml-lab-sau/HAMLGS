function [graph_euclidean, graph_cosine, graph_kernel] = knn_graphs(X, k)
    % Input:
    %   X: Data matrix (each row corresponds to a data point)
    %   k: Number of nearest neighbors
    
    % Euclidean Distance Graph
    D_euclidean = pdist2(X, X, 'euclidean');
    [~, idx_euclidean] = sort(D_euclidean, 2);
    graph_euclidean = zeros(size(D_euclidean));
    for i = 1:size(X, 1)
        graph_euclidean(i, idx_euclidean(i, 2:k+1)) = 1;  % Connect to k-nearest neighbors
    end

    % Cosine Similarity Graph
    X_normalized = X ./ sqrt(sum(X.^2, 2));
    cosine_similarity = X_normalized * X_normalized';
    [~, idx_cosine] = sort(cosine_similarity, 2, 'descend');
    graph_cosine = zeros(size(cosine_similarity));
    for i = 1:size(X, 1)
        graph_cosine(i, idx_cosine(i, 2:k+1)) = 1;  % Connect to k-nearest neighbors
    end

    % Kernel Matrix Graph (e.g., Gaussian RBF Kernel)
    sigma = 1;  % Adjust this parameter based on your requirements
    kernel_matrix = exp(-pdist2(X, X, 'euclidean').^2 / (2 * sigma^2));
    [~, idx_kernel] = sort(kernel_matrix, 2, 'descend');
    graph_kernel = zeros(size(kernel_matrix));
    for i = 1:size(X, 1)
        graph_kernel(i, idx_kernel(i, 2:k+1)) = 1;  % Connect to k-nearest neighbors
    end
end
