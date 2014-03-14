function H = mla_logistic_regression_sigmoid(z)
%% Purpose: given z, compute g(z)
%% Info: this function is fully vectorized

% convert into diag for computation convenience
Z_converted = diag(z);

% sigmoid function below
H = 1 ./ ( 1 + sum(e ^ (-Z_converted) , 2) ) ;

end
