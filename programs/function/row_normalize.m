function normalized_graph = row_normalize(graph)
    % Row-normalize the graph adjacency matrix
    row_sums = sum(graph, 2);
    normalized_graph = graph ./ max(row_sums, eps);
end
