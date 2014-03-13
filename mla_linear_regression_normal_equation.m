function theta = mla_linear_regression_normal_equation(X, y)
%% Purpose:	This function is to compute theta
%% Purpose: Normal equation algorithm is used here
%% Attention: if n > 10000, please start considering using gradient descent
%% Attention: if there is redundant features, X'*X may be non-invertible
%% Attention: if m<=n, X'*X may be non-invertible


theta = pinv(X' * X) * X' * y;

end
