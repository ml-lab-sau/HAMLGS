
function [F,F_test,time]=myfunconsensus(X,Y,X_test,par)
tic;
c2=par.c2;
c3=par.c3;
c4=par.c4;
c5=par.c5;
c6=par.c6;
lambda2=par.lambda2;
 theta = rand();
 tolerance=0.001;
 [d,n]=size(X);
c=size(Y,2);
U = zeros(n);
large_value = 1e10;  % Choose a suitable large value

% Set U values based on labels
for i = 1:n
    if any(Y(i, :) ~= 0)
        U(i,i) = large_value;
    else
        U(i,i) = 0;
    end
end
lambda=0.1;
t_values=[0.1,1,10];
% Construct graphs using KNN approach
k1 = 10;%floor(0.2*d);
[graph_euclidean, graph_cosine, graph_kernel] = knn_graphs(X', k1);
graph_kernel1=heat_kernel_graphs(X', t_values);
%function v = featureSelection(X, A, lambda1, lambda2)
Ast = struct();

    % Normalize A matrices
    Ast.A1=row_normalize(graph_euclidean);
    Ast.A2=row_normalize(graph_cosine);
    Ast.A3=row_normalize(graph_kernel1{1});
    Ast.A4=row_normalize(graph_kernel1{2});
    Ast.A5=row_normalize(graph_kernel1{3});
    %A = normalizeGraphs(A);
%function A_norm = normalizeGraphs(A)
    fieldnamesA = fieldnames(Ast);
    
    % Initialize A_norm with the first matrix
    A_norm = Ast.(fieldnamesA{1});
    
    % Add up the matrices along the third dimension
 for k = 2:numel(fieldnames(Ast))
    field = fieldnamesA{k};
    A_norm = A_norm + Ast.(field);
 end
A_avg = A_norm/numel(fieldnames(Ast));
% k1 = 10;
% [graph_euclidean, graph_cosine, graph_kernel] = knn_graphs(X', k1);
% %function v = featureSelection(X, A, lambda1, lambda2)
% Ast = struct();
% 
%     % Normalize A matrices
%     Ast.A1=row_normalize(graph_euclidean);
%     Ast.A2=row_normalize(graph_cosine);
%     Ast.A3=row_normalize(graph_kernel);
%     %A = normalizeGraphs(A);
% %function A_norm = normalizeGraphs(A)
%     fieldnamesA = fieldnames(Ast);
%     
%     % Initialize A_norm with the first matrix
%     A_norm = Ast.(fieldnamesA{1});
%     
%     % Add up the matrices along the third dimension
%  for k = 2:numel(fieldnames(Ast))
%     field = fieldnamesA{k};
%     A_norm = A_norm + Ast.(field);
%  end
% A_avg = A_norm/numel(fieldnames(Ast));
   epsilon=1e-6;
    % Initialize variables
    %n = size(X, 2);
    L= calculateLaplacian(A_avg);
    V=(1/d)*eye(d,d);
    F=ones(n,c);
    Q=ones(d,c);
      
    maxiter=100;
    AK=A_avg;
    %% Main loop until convergence
    %while notConverged()
    for i=1:maxiter
     % Step 4: Compute Q
     %Compute R (eigen vector of L)
     % Assuming Q is your matrix and epsilon is a small positive constant
       Q_norm_squared = sum(Q.^2, 2);
       numerator = sum(sqrt(Q_norm_squared + epsilon));
      denominator = sqrt(sum(Q.^2, 'all') + epsilon);
       R = numerator / denominator;

      % Build the coefficient matrix
      A = 2* X * L * X' + c4 * X * X' + c2 * diag(V.^(-1)) + lambda*X * X' + c6 * R*eye(d);

    % Right-hand side vector
      b = c4 * X * F;

% Solve for Q using the backslash operator or a solver of your choice
     Q = A \ b;
%       R=computeR(L,c);
%       Q = ((c5+1) * X * X' + c2 * diag(V).^(-1)) \ (c5 * X * F + X * R);
      % Step 5: Compute V
      V_diag = sqrt(sum(Q.^2, 2)) ./ sqrt(sum(sqrt(sum(Q.^2, 2))));
      V=diag(V_diag);
      %compute alpha
      alpha = find_alpha(AK, Ast);
       % Step 6: Compute A
         % Generates a random value between 0 and 1   
          M=calculate_M(Ast,AK,alpha);
         AK = optimizeA(Q, X, c6, AK,theta, alpha,M);
         %disp(AK);
       % Identity matrix
       L=calculateLaplacian(AK);
       % Identity matrix
     % Identity matrix
      I = eye(size(U));

    % Calculate the matrix F 
    % F = (c3 * Y * U - 2 * c5 * Y - 2 * c5 * Q' * X)*(c3 * U + c4 * L + c5 * I)^-1;
   % F=(c5*Q'*X+c3*Y*U)*(c4*L+c5*I+c3*U)^-1;
    matrix_expression = (c3 * U  + c5 *I)^-1;
    F = matrix_expression* (c3 * U * Y + c5 * X' * Q);
        %% Check for convergence (you need to implement this function)
        new_obj_value = compute_objective(Q, F, Ast, alpha, c2, c3, c4, c5, c6, lambda2, X, Y, U, V,L,AK);
        % Display or return any other relevant results
        if i > 1 && abs(new_obj_value - obj_value(i-1)) < tolerance
        %disp('Converged!');
        break;  % Exit the loop
    end

    obj_value(i) = new_obj_value;
    
    end
    F_test=sign(X_test'*Q);
    time=toc;
end


%% Function section
% Function to calculate laplacian
function L = calculateLaplacian(A)
    % Calculate the Degree Matrix
    D = diag(sum(A + A', 2) / 2);

    % Avoid division by zero by adding a small constant
    epsilon = 1e-10;
    D_inv_sqrt = diag(1 ./ sqrt(max(sum(A + A', 2), epsilon)));

    % Calculate the Laplacian Matrix
    L = D - (A + A') / 2;
end
%% Function to calculate Adjacency Matrix
function A = optimizeA(Q, X, c6,A_avg,theta, alpha,M)
     n = size(X, 2);  % Assuming X is a d x n matrix
    % Calculate B and M matrices
    B=calculate_B(Q, X);
   % Update A using KKT conditions
    for i = 1:n
        for j = 1:n
            if i ~= j && M(i, j) ~= 0
                A_avg(i, j) = c6 * M(i, j) / (B(i, j) + theta);
            end
        end
    end
A=A_avg;

% Row-normalize A_avg
    A = bsxfun(@rdivide, A_avg, sum(A_avg, 2));
end
%% Function to calculate R(Eigen vector of Laplacian)
function R=computeR(L,c)
[EV, ED] = eig(L);
R = EV(:, 1:c);
R = R / norm(R);
end
%% Function to calculate alpha 
function alpha = find_alpha(AK,Ast)
    % Input:
    %   AK: Reference adjacency matrix
    %   Ast: Structure with fields containing individual adjacency matrices

    % Extract the number of graphs
    m = numel(fieldnames(Ast));

    % Calculate M_k matrices for each graph
    M_values = zeros(1, m);
    for k = 1:m
        field_name = ['A', num2str(k)];
        if isfield(Ast, field_name)
            A_k = Ast.(field_name);
            M_values(k) = sum(sum(A_k .* log((A_k + eps) ./ (AK + eps)), 'omitnan'), 'omitnan');
        else
            error(['Field "', field_name, '" is missing in the Ast structure.']);
        end
    end

    % Compute inverse of M_k matrices, handling potential division by zero
    inv_M_values = 1 ./ max(M_values, eps);

    % Calculate the sum of inverses
    sum_inv_M = sum(inv_M_values);

    % Normalize the alphas to ensure the sum is exactly 1
    alpha = inv_M_values / sum_inv_M;
end

%% Function to calculate B 
function B = calculate_B(Q, X)
    % Calculate B matrix
    n = size(X, 2);
    B = zeros(n, n);
    for i = 1:n
        for j = 1:n
            B(i, j) = norm(Q' * (X(:, i) - X(:, j)))^2;
        end
    end
end
%% Function to calculate M
function M = calculate_M(Ast,Ak,alpha)
    % Calculate M matrix element-wise for all graphs in the structure
    m = numel(alpha);
    
    % Get the size of the adjacency matrix in the structure
    [n, ~] = size(Ast.(['A', num2str(1)]));
    
    % Initialize M with zeros
    M = zeros(n);
    
    % Iterate through each graph
    for k = 1:m
        % Get the A matrix for the current graph
        A_current = Ast.(['A', num2str(k)]);
        
        % Calculate the M matrix element-wise for the current graph
        M = M + alpha(k)^2 * A_current;
    end
end
%% Function to calculate objective function value to check for convergence
function obj_value = compute_objective(Q, F, Ast, alpha, c2, c3, c4, c5, c6, lambda2, X, Y, U, V, L, AK)
    % Number of graphs
       m = numel(alpha);
    eps_value = 1e-15;
    % Data fidelity term
    data_fidelity = 0;
    for k = 1:m
        % Check if the field 'A{k}' exists in the structure Ast
        field_name = ['A', num2str(k)];
        if isfield(Ast, field_name)
            A_k = Ast.(field_name);  % Get the matrix A{k} from the structure
            data_fidelity = data_fidelity + alpha(k)^2 * sum(sum(A_k .* log((A_k+eps_value) ./( AK+eps_value))));
        else
            % Handle the case where the field is missing
            warning(['Field "', field_name, '" is missing in the Ast structure. Skipping this term.']);
        end
    end

   % Regularization term 1
    reg_term_1 = trace(Q'*X*L*X'*Q)+c2 * trace(Q' * inv(V) * Q);  % Assuming V is a square invertible matrix
% 
%     % Regularization term 2
    reg_term_2 = c3 * trace((F - Y)' * U * (F - Y));
% 
%     % Regularization term 3
    %reg_term_3 = c4 * trace(F' * L * F);
% 
%     % Data fitting term
    data_fitting = c5 * norm( F-X'*Q, 'fro')^2;
% 
%     % Graph regularization term
    graph_reg = c6 * lambda2 * data_fidelity;

    % Total objective function value
    obj_value =  reg_term_1 + reg_term_2 + data_fitting + graph_reg;
end



