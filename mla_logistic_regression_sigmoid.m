function H = mla_logistic_regression_sigmoid(z)
%% Purpose: given z, compute g(z)
%% Info: this function is fully vectorized

% sigmoid function below
H = 1 ./ ( 1 + e .^ (-z) ) ;

end
