function shannon_entropy = compute_shannon_entropy(data, selected_features)
    % Compute Shannon entropy for the selected features
 
    p = sum(selected_features) / numel(selected_features);
    r = 1 - p;
    
    % Handling the case where p or q is zero
    p(p == 0) = 1; % Set p to 1 to avoid log(0)
    r(r == 0) = 1; % Set q to 1 to avoid log(0)
    
    entropy = -p .* log2(p) - r .* log2(r);
    shannon_entropy = sum(entropy);
end
