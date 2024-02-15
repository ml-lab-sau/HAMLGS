% function mi_scores = compute_mutual_information(data, class_size)
%     num_features = size(data, 2);
%     mi_scores = zeros(1, num_features);
%     
%     for i = 1:num_features
%         % Create histograms for the feature and the class variable
%         [feature_counts, ~] = histcounts(data(:, i), 'Normalization', 'probability');
%         
%         % Ensure class_size is a column vector for compatibility
%         class_size = class_size(:);  % Convert to a column vector
%         [class_counts, edges] = histcounts(class_size, 'Normalization', 'probability');
%         
%         % Compute joint histogram for the feature and class
%         joint_counts = histcounts2(data(:, i), class_size, edges, edges, 'Normalization', 'probability');
%         
%         % Calculate mutual information
%         mi = 0;
%         for j = 1:numel(feature_counts)
%             for k = 1:numel(class_counts)
%                 if joint_counts(j, k) > 0
%                     mi = mi + joint_counts(j, k) * log(joint_counts(j, k) / (feature_counts(j) * class_counts(k)));
%                 end
%             end
%         end
%         mi_scores(i) = mi;
%     end
% end
function q = calculate_mutual_information_matrix(data)
    d = size(data, 2);
    q = zeros(d, d);

    for i = 1:d
        for j = i+1:d
            % Calculate mutual information between feature i and feature j
            mi = estimate_mutual_information(data(:, i), data(:, j));

            % Store the mutual information value in the matrix
            q(i, j) = mi;
            q(j, i) = mi; % Mutual information is symmetric
        end
    end
end

function mi = estimate_mutual_information(x, y)
    % Estimate mutual information between variables x and y using histograms

    % Number of bins for the histogram
    num_bins = 10;

    % Compute histograms
    hist_x = histcounts(x, num_bins);
    hist_y = histcounts(y, num_bins);
    hist_xy = histcounts2(x, y, [num_bins, num_bins]);

    % Normalize histograms
    p_x = hist_x / sum(hist_x);
    p_y = hist_y / sum(hist_y);
    p_xy = hist_xy / sum(hist_xy(:));

    % Calculate mutual information
    mi = 0;
    for i = 1:num_bins
        for j = 1:num_bins
            if p_xy(i, j) > 0
                mi = mi + p_xy(i, j) * log2(p_xy(i, j) / (p_x(i) * p_y(j)));
            end
        end
    end
end

