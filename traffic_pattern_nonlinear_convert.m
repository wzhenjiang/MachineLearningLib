function poly_matrix = traffic_pattern_nonlinear_convert(X, n)
%% Purpose: this function is to add high power features of X

poly_matrix = X;

seed = poly_matrix(:,end);

poly_matrix = [poly_matrix(:,1:end-1), mla_mean_normalization(seed, [],[],true)];
for i = 1 : n-1
	poly_matrix = [poly_matrix, mla_mean_normalization(poly_matrix(:,end) .* seed, [], [], true)];
end

end