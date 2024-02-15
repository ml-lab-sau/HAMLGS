% Example: Constructing hyperedges using heat kernel similarity
function H =HyperedgeHeatKarnel(data_points)

% Compute pairwise Euclidean distances (you can use other similarity measures)
pairwise_distances = pdist2(data_points, data_points);

% Set a threshold for the heat kernel similarity
threshold = 0.5;  % Adjust based on your specific problem

% Compute heat kernel similarity matrix
heat_kernel_similarity = exp(-pairwise_distances.^2 / (2 * threshold^2));

num_hyperedges = size(data_points,1);

% Define hyperedges based on the heat kernel similarity
hyperedges = cell(size(heat_kernel_similarity, 1), 1);
for i = 1:size(heat_kernel_similarity, 1)
    hyperedges{i} = find(heat_kernel_similarity(i, :) > 0);  % Vertices with similarity above threshold
end

% Display the hyperedges
% disp('Hyperedges:');
% disp(hyperedges);
for i = 1:num_hyperedges
    vertices_in_hyperedge = hyperedges{i};
    incidence_matrix(i, vertices_in_hyperedge) = 1;
end


H=incidence_matrix;

 end
