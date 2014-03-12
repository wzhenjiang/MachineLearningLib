function theta = mla_normal_equation(X, y)
%% Purpose:	This function is to compute theta
%% Purpose: Normal equation algorithm is used here

theta = pinv(X' * X) * X' * y;

end