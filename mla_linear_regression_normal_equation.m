function theta = mla_linear_regression_normal_equation(X, y, lambda = 0)
%% Purpose:	This function is to compute theta
%% Purpose: Normal equation algorithm is used here
%% Info: lambda is used for regularization purpose. default 0 means no regularization
%% Attention: if n > 10000, please start considering using gradient descent
%% Attention: if there is redundant features, X'*X may be non-invertible
%% Attention: if m<=n, X'*X may be non-invertible

% all below are used for regularization
[m,n] = size(X);
wierd_matrix = eye(n,n);
wierd_matrix(1,1) = 0;

% regularization is adopted in lambda * wierd_matrix
theta = pinv(X' * X + lambda * wierd_matrix) * X' * y;

end
