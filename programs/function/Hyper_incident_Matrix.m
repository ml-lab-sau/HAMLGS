% Example: Constructing hyperedges using t-SNE for manifold learning
function H=Hyper_incident_Matrix(data_points,K)
% Sample data points (vertices)
[num_vertices,num_dim] = size(data_points);
%data_points = rand(num_vertices, 10);  % Random 10D data

% Perform t-SNE to embed the data into 2D
embedding_dim = floor(0.2*num_dim);
tsne_embeddings = tsne(data_points, 'NumDimensions', embedding_dim);

% Set a threshold for grouping vertices into hyperedges
threshold =K; % 5.0;  % Adjust based on your specific problem

% Compute pairwise distances in the embedded space
pairwise_distances = pdist2(tsne_embeddings, tsne_embeddings);

% Define hyperedges based on distance threshold
hyperedges = cell(num_vertices, 1);
for i = 1:num_vertices
    hyperedges{i} = find(pairwise_distances(i, :) < threshold);
end

% Display the hyperedges
%disp('Hyperedges:');
%disp(hyperedges);
% Initialize the incidence matrix
num_hyperedges = num_vertices;
num_vertices = size(tsne_embeddings, 1);
incidence_matrix = zeros(num_hyperedges, num_vertices);

% Construct the incidence matrix based on hyperedges
for i = 1:num_hyperedges
    vertices_in_hyperedge = hyperedges{i};
    incidence_matrix(i, vertices_in_hyperedge) = 1;
end

% Display the incidence matrix
%disp('Incidence Matrix:');
%disp(incidence_matrix);
H=incidence_matrix;
end
