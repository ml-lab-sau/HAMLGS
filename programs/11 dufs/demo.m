clear; clc;
load('wine.mat');
% class_size=size(target,2);
% lambda1=0.1;
% lambda2=0.2;
% lambda3=1;
% Sample confusion matrix (replace with your own confusion matrix)
% Rows: true labels, Columns: predicted labels
confusion_matrix = [
    50  5  5;   % Class 1
    10 45  5;   % Class 2
    5   5 50    % Class 3
];

% Calculate TP, FP, FN, and TN for each class
TP = diag(confusion_matrix);
FP = sum(confusion_matrix, 1) - TP;
FN = sum(confusion_matrix, 2) - TP;
TN = sum(confusion_matrix(:)) - TP - FP - FN;

% Calculate precision, recall, and F1 score for each class
precision = TP ./ (TP + FP);
recall = TP ./ (TP + FN);
f1_score_per_class = 2 * (precision .* recall) ./ (precision + recall);

% Calculate Macro F1 score (average F1 score across classes)
macro_f1_score = mean(f1_score_per_class);

% Calculate Micro F1 score (aggregate TP, FP, FN, across all classes)
micro_precision = sum(TP) / sum(TP + FP);
micro_recall = sum(TP) / sum(TP + FN);
micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall);

disp('F1 Score per class:');
disp(f1_score_per_class);
disp('Macro F1 Score:');
disp(macro_f1_score);
disp('Micro F1 Score:');
disp(micro_f1_score);

feature_idx = dufs(data, class_size, lambda1, lambda2, lambda3);